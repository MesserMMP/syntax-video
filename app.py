# app.py — минимальный UI: загрузка всех DICOM → classify → per-side inference

import os
import base64
import json
import tempfile
from datetime import datetime
from dataclasses import asdict
from typing import List, Dict, Any

import gradio as gr
import pydicom

from src.syntax_pred.config import CFG
from src.syntax_pred.infer import Study, run_inference

# ------- Логотип (base64) -------
DEFAULT_LOGO = "assets/logo.png"
LOGO_PATH = os.environ.get("LOGO_PATH", DEFAULT_LOGO)


def _logo_html() -> str:
    path = LOGO_PATH
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            import base64 as b64
            data = b64.b64encode(f.read()).decode("ascii")
        ext = os.path.splitext(path)[1].lower()
        mime = "image/png" if ext in {".png", ""} else "image/jpeg"
        return (
            f'<img src="data:{mime};base64,{data}" alt="logo" '
            f'style="height:40px;vertical-align:middle;display:inline-block;'
            f'image-rendering:auto;object-fit:contain;margin-right:12px;" />'
        )
    except Exception:
        return ""


# ------- Вспомогательные -------

def _is_dicom_path(path: str) -> bool:
    """Фильтрация только настоящих DICOM (по расширению и попытке чтения)."""
    if not os.path.exists(path):
        return False
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".dcm", ""}:  # многие DICOM без расширения
        return False
    try:
        pydicom.dcmread(path, stop_before_pixels=True)
        return True
    except Exception:
        return False


def _files_to_paths(files) -> List[str]:
    raw_paths = [f.name for f in (files or []) if hasattr(f, "name") and os.path.exists(f.name)]
    return [p for p in raw_paths if _is_dicom_path(p)]


def _collect_input_paths(files_all, folder_files) -> List[str]:
    combined = _files_to_paths(files_all) + _files_to_paths(folder_files)
    # Keep first occurrence order while removing duplicates.
    return list(dict.fromkeys(combined))


def _build_report_file(result: Dict[str, Any]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f"syntax_report_{ts}_",
        suffix=".json",
        encoding="utf-8",
        delete=False,
    ) as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        return f.name


def _status_badge(state: str) -> str:
    """Красивый статус с «анимацией» точек через Unicode."""
    state = state.lower()
    if state == "running":
        # Используем точки/круги как псевдо-анимацию
        return "⏱️ Running · ● ○ ○"
    if state == "done":
        return "✅ Done"
    if state == "error":
        return "❌ Error"
    return "⌛ Queued"


def _update_status_table(studies, status: str):
    """Пометить все исследования указанным статусом (только визуальный бейдж)."""
    badge = _status_badge(status)
    return [
        [s["name"], s.get("description", ""), len(s.get("files", [])), badge]
        for s in (studies or [])
    ]


def _format_results_html(result: Dict[str, Any]) -> str:
    if not result:
        return (
            '<div style="padding:12px;border:1px solid #e2e8f0;border-radius:10px;">'
            '<b>Результаты появятся после запуска inference.</b>'
            '</div>'
        )

    if "error" in result:
        msg = str(result.get("error", "Unknown error"))
        return (
            '<div style="padding:12px;border:1px solid #fecaca;background:#fef2f2;'
            'border-radius:10px;color:#991b1b;">'
            f'<b>Ошибка:</b> {msg}'
            '</div>'
        )

    studies = result.get("studies", []) or []
    if not studies:
        return (
            '<div style="padding:12px;border:1px solid #e2e8f0;border-radius:10px;">'
            'Нет исследований для отображения.'
            '</div>'
        )

    cards = []
    for s in studies:
        study = s.get("study", "-")
        desc = s.get("description", "")

        left_mean = s.get("left", {}).get("mean", 0.0)
        right_mean = s.get("right", {}).get("mean", 0.0)

        total_obj = s.get("total", {})
        total_mean = total_obj.get("mean", 0.0)

        risk_key = next((k for k in total_obj.keys() if "High-risk" in str(k)), "High-risk")
        is_high_risk = bool(total_obj.get(risk_key, False))

        badge_bg = "#fee2e2" if is_high_risk else "#dcfce7"
        badge_fg = "#991b1b" if is_high_risk else "#166534"
        badge_text = "Высокий риск" if is_high_risk else "Низкий риск"

        desc_html = f"<div style='color:#64748b;margin-top:4px'>{desc}</div>" if desc else ""

        cards.append(
            """
            <div style="border:1px solid #e2e8f0;border-radius:12px;padding:14px 16px;margin-bottom:10px;background:#ffffff;">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">
                <div>
                  <div style="font-size:18px;font-weight:700;color:#0f172a;">{study}</div>
                  {desc_html}
                </div>
                <div style="padding:4px 10px;border-radius:999px;background:{badge_bg};color:{badge_fg};font-weight:700;white-space:nowrap;">
                  {badge_text}
                </div>
              </div>
              <div style="margin-top:10px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;">
                <div style="background:#f8fafc;border-radius:10px;padding:10px;">
                  <div style="color:#64748b;font-size:12px;">LEFT</div>
                  <div style="font-size:22px;font-weight:800;color:#0f172a;">{left_mean:.3f}</div>
                </div>
                <div style="background:#f8fafc;border-radius:10px;padding:10px;">
                  <div style="color:#64748b;font-size:12px;">RIGHT</div>
                  <div style="font-size:22px;font-weight:800;color:#0f172a;">{right_mean:.3f}</div>
                </div>
                <div style="background:#f1f5f9;border-radius:10px;padding:10px;border:1px solid #cbd5e1;">
                  <div style="color:#334155;font-size:12px;">TOTAL</div>
                  <div style="font-size:24px;font-weight:900;color:#020617;">{total_mean:.3f}</div>
                </div>
              </div>
            </div>
            """.format(
                study=study,
                desc_html=desc_html,
                badge_bg=badge_bg,
                badge_fg=badge_fg,
                badge_text=badge_text,
                left_mean=left_mean,
                right_mean=right_mean,
                total_mean=total_mean,
            )
        )

    return "\n".join(cards)


# ------- UI -------
def ui():
    with gr.Blocks() as demo:
        gr.HTML(
            f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                {_logo_html()}
                <h1 style="margin:0;font-weight:800;text-align:center;flex:1;">SYNTAX-Video — Study Inference</h1>
            </div>
            <ol style="margin:0 0 12px 20px; color:#475569; line-height:1.5;">
              <li>Укажите ID исследования и (необязательно) описание.</li>
              <li>Загрузите все DICOM-файлы исследования одним списком или папкой.</li>
              <li>Нажмите “Add study”, затем “Run inference”.</li>
            </ol>
            """
        )

        studies_state = gr.State([])  # list[dict]

        with gr.Row():
            study_name = gr.Textbox(label="Study ID", placeholder="e.g., S1234")
            study_desc = gr.Textbox(label="Description (optional)", placeholder="Free text...")

        files_all = gr.File(label="All DICOM files (single study)", file_count="multiple")
        files_folder = gr.File(label="Study folder (optional)", file_count="directory")

        with gr.Row():
            btn_add = gr.Button("➕ Add study")
            btn_clear = gr.Button("🗑️ Clear all")

        queue_table = gr.Dataframe(
            headers=["Study", "Description", "#DICOM files", "Status"],
            datatype=["str", "str", "number", "str"],
            interactive=False,
            label="Studies queue",
            row_count=(0, "dynamic"),
        )

        def _add_study_fn(studies: List[Dict[str, Any]], name, desc, files, folder):
            name = (name or "").strip() or f"Study_{len(studies)+1}"
            desc = (desc or "").strip()
            paths = _collect_input_paths(files, folder)
            if not paths:
                # ничего не добавляем, просто обновляем таблицу текущими статусами
                table = _update_status_table(studies, "Queued")
                return studies, table, name, desc, files, folder

            studies = studies + [asdict(Study(name=name, description=desc, files=paths))]
            table = _update_status_table(studies, "Queued")
            return studies, table, "", "", None, None

        btn_add.click(
            _add_study_fn,
            inputs=[studies_state, study_name, study_desc, files_all, files_folder],
            outputs=[studies_state, queue_table, study_name, study_desc, files_all, files_folder],
        )

        def _clear_all():
            return [], []

        btn_clear.click(_clear_all, inputs=None, outputs=[studies_state, queue_table])

        run_btn = gr.Button("🚀 Run inference", variant="primary")
        out_summary = gr.HTML(label="Результат")
        with gr.Accordion("Подробно (JSON)", open=False):
            out_json = gr.JSON(label="Results")
        out_report = gr.File(label="Скачать отчет", interactive=False)

        def _before_run(studies):
            # пометить все исследования как Running
            return _update_status_table(studies, "Running")

        def _run_infer(studies):
            study_objs = [Study(**s) for s in (studies or [])]
            result = run_inference(study_objs)
            report_path = _build_report_file(result)
            return result, report_path

        def _after_run(studies, result, report_path):
            table = _update_status_table(studies, "Done")
            return _format_results_html(result), result, report_path, table

        run_btn.click(
            _before_run,
            inputs=[studies_state],
            outputs=[queue_table],
        ).then(
            _run_infer,
            inputs=[studies_state],
            outputs=[out_json, out_report],
        ).then(
            _after_run,
            inputs=[studies_state, out_json, out_report],
            outputs=[out_summary, out_json, out_report, queue_table],
        )

        gr.Markdown("⚠️ Research-only. Not a medical device. Predictions depend on input quality and domain shift.")
    return demo


if __name__ == "__main__":
    favicon = LOGO_PATH if (LOGO_PATH and os.path.exists(LOGO_PATH)) else None
    ui().launch(favicon_path=favicon, theme=gr.themes.Soft())
