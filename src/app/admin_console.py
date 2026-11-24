"""Operations console for AgriSight-BD.

Admins land here instead of the field planner. A season uploaded or
opened on this console becomes the latest year the planner uses.
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from src.accounts import (
    REVIEW_CATEGORIES,
    ROLE_LABELS,
    application_document_bytes,
    assign_review_request,
    close_review_request,
    complete_review_request,
    list_agronomist_applications,
    list_review_events,
    list_review_requests,
    list_verified_agronomists,
    review_agronomist_application,
    start_review_request,
)
from src.config import CROP_CATALOG
from src.year_intake import (
    apply_crop_rows,
    apply_greenness,
    apply_soil,
    apply_upload,
    apply_weather,
    catalog_frame,
    crop_file_bytes,
    crop_year_csv,
    datasets_zip_bytes,
    greenness_template_csv,
    open_seasons,
    parse_crop_rows,
    parse_greenness,
    parse_soil,
    parse_upload,
    parse_weather,
    registry_stamp,
    score_season,
    season_status,
    soil_template_csv,
    template_csv,
    weather_template_csv,
)


ADMIN_PAGES = [
    "Applications",
    "Operations",
    "Year intake",
    "Season results",
    "Crop files",
    "Expert review",
]
AGRONOMIST_PAGES = ["Expert review"]

ADMIN_CSS = """
<style>
[data-testid="stAppViewContainer"]:has(.admin-shell) {
    background: #E7ECF1;
}
[data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stHeader"] {
    display: none !important;
}
[data-testid="stAppViewContainer"]:has(.admin-shell) .block-container {
    max-width: 1240px;
    padding-top: 1.2rem;
}
[data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stSidebar"] {
    background: #12181F;
    border-right: 1px solid #243140;
}
[data-testid="stAppViewContainer"]:has(.admin-shell) .brand {
    border-bottom-color: #2A3A4A;
}
[data-testid="stAppViewContainer"]:has(.admin-shell) .brand .mark {
    color: #E0B15A;
}
[data-testid="stAppViewContainer"]:has(.admin-shell) .brand h1 {
    color: #F7F4EF !important;
}
[data-testid="stAppViewContainer"]:has(.admin-shell) .brand p {
    color: #9AABBC !important;
}
[data-testid="stAppViewContainer"]:has(.admin-shell) .nav-label {
    color: #8EA0B5;
}
[data-testid="stAppViewContainer"]:has(.admin-shell) .account-pill {
    background: #1A2430;
    border-color: #2C3E50;
}
[data-testid="stAppViewContainer"]:has(.admin-shell) .account-pill span {
    color: #F4F7FA;
}
[data-testid="stAppViewContainer"]:has(.admin-shell) .account-pill p {
    color: #9AABBC;
}
[data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stSidebar"] button {
    justify-content: flex-start !important;
}
[data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stSidebar"] button p {
    text-align: left !important;
    width: 100%;
}
[data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
    background: transparent;
    border: 0;
    color: #D5DEE8;
    text-align: left;
    justify-content: flex-start;
    box-shadow: none;
}
[data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
    background: #243447;
    border: 0;
    color: #F8FBFF;
    text-align: left;
    justify-content: flex-start;
    border-radius: 8px;
    box-shadow: inset 3px 0 0 #D4A017;
}
[data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stMain"] button[data-testid="stBaseButton-primary"] {
    background: #1F3A5F;
    border: 1px solid #1F3A5F;
    color: #FFFFFF;
}
[data-testid="stAppViewContainer"]:has(.admin-shell) h2,
[data-testid="stAppViewContainer"]:has(.admin-shell) h3 {
    color: #14202B !important;
}
.ops-head {
    margin-bottom: 14px;
}
.ops-head .eyebrow {
    color: #8A6A24;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 0 0 6px 0;
}
.ops-head h1 {
    color: #14202B !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.15 !important;
    margin: 0 0 8px 0 !important;
}
.ops-head p {
    color: #3E4E5E !important;
    font-size: 1.02rem !important;
    line-height: 1.5 !important;
    margin: 0 !important;
    max-width: 760px;
}
.ops-kpis, .ops-steps {
    display: grid;
    gap: 12px;
    margin: 16px 0;
}
.ops-kpis {
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
}
.ops-kpis.review-board {
    grid-template-columns: repeat(3, minmax(0, 1fr));
}
.ops-steps {
    grid-template-columns: repeat(3, minmax(0, 1fr));
}
.ops-kpis div, .ops-steps div {
    background: #FFFFFF;
    border: 1px solid #D5DEE6;
    border-radius: 12px;
    padding: 14px 16px 12px;
}
.ops-kpis div {
    border-top: 3px solid #1F3A5F;
}
.ops-kpis span, .ops-steps span {
    display: block;
    color: #5C6E80;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.ops-steps span {
    color: #8A6A24;
    letter-spacing: 0.12em;
}
.ops-kpis strong, .ops-steps strong {
    display: block;
    color: #14202B;
    margin-top: 4px;
}
.ops-kpis strong {
    font-size: 1.55rem;
    letter-spacing: -0.03em;
    line-height: 1.15;
}
.apply-card {
    background: #FFFFFF;
    border: 1px solid #D5DEE6;
    border-radius: 14px;
    padding: 16px 16px 12px;
    margin: 0 0 12px 0;
}
.apply-card h3 {
    margin: 0 0 4px 0 !important;
    font-size: 1.05rem !important;
}
.ops-kpis em, .ops-steps p {
    display: block;
    color: #3E4E5E;
    font-style: normal;
    font-size: 0.88rem;
    line-height: 1.45;
    margin: 4px 0 0 0;
}
.ops-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 4px 0 8px 0;
}
.ops-pills span {
    background: #FFFFFF;
    border: 1px solid #D5DEE6;
    border-radius: 999px;
    padding: 6px 12px;
    color: #1F3A5F;
    font-size: 0.82rem;
    font-weight: 650;
}
.reviewer-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin: 0 0 16px 0;
}
.reviewer-row span {
    display: block;
    background: #FFFFFF;
    border: 1px solid #D5DEE6;
    border-left: 3px solid #1F3A5F;
    border-radius: 12px;
    padding: 12px 14px;
    color: #14202B;
    font-size: 0.95rem;
    font-weight: 700;
}
.reviewer-row em {
    display: block;
    margin-top: 3px;
    color: #5C6E80;
    font-style: normal;
    font-size: 0.78rem;
    font-weight: 650;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
@media (max-width: 1200px), (pointer: coarse) {
    [data-testid="stAppViewContainer"]:has(.admin-shell) {
        overflow-x: hidden !important;
    }
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stHeader"] {
        display: block !important;
        height: 0 !important;
        min-height: 0 !important;
        background: transparent !important;
        border: 0 !important;
        overflow: visible !important;
    }
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stSidebarHeader"],
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stSidebarCollapsedControl"],
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="collapsedControl"] {
        display: flex !important;
        position: fixed !important;
        top: calc(18px + env(safe-area-inset-top)) !important;
        z-index: 5000 !important;
        width: 32px !important;
        height: 32px !important;
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
    }
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarHeader"] {
        left: calc(min(86vw, 280px) - 44px) !important;
    }
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] {
        color: #F4F7FA !important;
    }
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stSidebarCollapsedControl"],
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="collapsedControl"] {
        left: 14px !important;
        background: #FFFFFF !important;
        border: 1px solid #D5DEE6 !important;
        border-radius: 8px !important;
    }
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {
        color: #14202B !important;
    }
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stAppDeployButton"],
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stMainMenuButton"],
    [data-testid="stAppViewContainer"]:has(.admin-shell) #MainMenu {
        display: none !important;
    }
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stSidebar"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        margin-left: 0 !important;
        width: min(86vw, 280px) !important;
        min-width: 0 !important;
        max-width: min(86vw, 280px) !important;
        height: 100dvh !important;
        max-height: 100dvh !important;
        z-index: 1000 !important;
        overflow: visible !important;
        transform: none !important;
        box-shadow: 0 16px 40px rgba(18, 24, 31, 0.28);
        transition: left 180ms ease, visibility 180ms ease;
    }
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stSidebar"][aria-expanded="false"] {
        left: -110% !important;
        transform: none !important;
        visibility: hidden !important;
    }
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stSidebar"][aria-expanded="true"] {
        left: 0 !important;
        transform: none !important;
        visibility: visible !important;
    }
    [data-testid="stAppViewContainer"]:has(.admin-shell) > div:has([data-testid="stMain"]) {
        width: 100% !important;
        max-width: 100% !important;
    }
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stMain"] {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
    }
    [data-testid="stAppViewContainer"]:has(.admin-shell) .block-container {
        max-width: 100% !important;
        width: 100% !important;
        padding: 4.2rem 0.9rem 2.2rem !important;
    }
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stColumn"] {
        flex: 1 1 100% !important;
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
    }
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.75rem !important;
    }
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stDataFrame"],
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stTable"],
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stFileUploader"] {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        overflow-x: auto !important;
    }
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stMain"] button {
        white-space: normal !important;
        height: auto !important;
        min-height: 2.6rem;
    }
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stExpander"] p,
    [data-testid="stAppViewContainer"]:has(.admin-shell) [data-testid="stMarkdownContainer"] {
        overflow-wrap: anywhere;
    }
    .ops-kpis, .ops-steps, .ops-kpis.review-board, .reviewer-row {
        grid-template-columns: 1fr 1fr;
    }
    .ops-head h1 {
        font-size: clamp(1.45rem, 6vw, 1.8rem) !important;
        overflow-wrap: anywhere;
    }
    .ops-head p,
    .ops-kpis em,
    .ops-pills span,
    .reviewer-row span {
        overflow-wrap: anywhere;
    }
}
@media (max-width: 640px) {
    .ops-kpis, .ops-steps, .ops-kpis.review-board, .reviewer-row {
        grid-template-columns: 1fr;
    }
    .ops-head p {
        font-size: 0.96rem !important;
    }
    [data-testid="stAppViewContainer"]:has(.admin-shell) .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }
}
</style>
"""


def _shell() -> None:
    st.markdown(ADMIN_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="admin-shell" style="display:none">operations</div>',
        unsafe_allow_html=True,
    )


def _head(title: str, subtitle: str = "") -> None:
    detail = (
        f"<p>{html.escape(subtitle)}</p>"
        if subtitle
        else ""
    )
    st.markdown(
        f"""
        <header class="ops-head">
            <p class="eyebrow">Operations</p>
            <h1>{html.escape(title)}</h1>
            {detail}
        </header>
        """,
        unsafe_allow_html=True,
    )


def _kpis(cards: list[tuple[str, str, str]], *, columns: int = 0) -> None:
    cells = []
    for label, value, note in cards:
        cells.append(
            "<div>"
            f"<span>{html.escape(label)}</span>"
            f"<strong>{html.escape(value)}</strong>"
            f"<em>{html.escape(note)}</em>"
            "</div>"
        )
    board = " review-board" if columns else ""
    st.markdown(
        f'<div class="ops-kpis{board}">{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )


def _flash() -> None:
    message = st.session_state.pop("ops_flash", None)
    if message:
        st.success(message)


def _refresh_planner() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()


def _open_planner() -> None:
    st.session_state.admin_console = False
    st.session_state.page = "Dashboard"


def _sign_out() -> None:
    st.session_state.pop("user_id", None)
    st.session_state.page = "Dashboard"
    st.session_state.admin_console = False


def _sidebar(user: dict) -> str:
    st.sidebar.markdown(
        """
        <div class="brand">
            <span class="mark">Operations</span>
            <h1>AgriSight</h1>
            <p>Season control for Bangladesh</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    role = ROLE_LABELS.get(user["role"], user["role"])
    st.sidebar.markdown(
        f"""
        <div class="account-pill">
            <span>{html.escape(user["name"])}</span>
            <p>{html.escape(role)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    pages = (
        ADMIN_PAGES
        if user["role"] == "admin"
        else AGRONOMIST_PAGES
    )
    default_page = "Operations" if user["role"] == "admin" else "Expert review"
    current = st.session_state.get("ops_page", default_page)
    if current not in pages:
        current = default_page
        st.session_state.ops_page = current
    st.sidebar.markdown(
        '<div class="nav-label">Console</div>',
        unsafe_allow_html=True,
    )
    for item in pages:
        if st.sidebar.button(
            item,
            key=f"ops_nav_{item}",
            type="primary" if item == current else "secondary",
            width="stretch",
        ):
            st.session_state.ops_page = item
            st.rerun()
    st.sidebar.markdown(
        '<div class="nav-label">Leave</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.button(
        "Open the planner",
        width="stretch",
        key="ops_open_planner",
        on_click=_open_planner,
    )
    st.sidebar.button(
        "Sign out",
        width="stretch",
        key="ops_sign_out",
        on_click=_sign_out,
    )
    return st.session_state.get("ops_page", "Operations")


def _stamp(value: str) -> str:
    return (value or "")[:16].replace("T", " ") + " UTC"


def _applications(user: dict) -> None:
    """Accept or deny agronomist identity applications."""

    _head("Agronomist applications")
    _flash()
    applications, error = list_agronomist_applications(user["id"])
    if error:
        st.error(error)
        return
    pending = [item for item in applications if item["status"] == "pending"]
    approved = [item for item in applications if item["status"] == "approved"]
    denied = [item for item in applications if item["status"] == "rejected"]
    _kpis(
        [
            ("Waiting", str(len(pending)), "Need a decision"),
            ("Accepted", str(len(approved)), "Verified agronomists"),
            ("Denied", str(len(denied)), "Can apply again"),
        ]
    )
    if not pending:
        st.caption("No applications waiting.")
    for item in pending:
        with st.container(border=True):
            st.markdown(f"**{item['full_name']}** · {item['email']}")
            st.caption(
                f"{item['organization']} · submitted "
                f"{_stamp(item['submitted_at'])}"
            )
            st.write(item["qualifications"])
            st.caption(
                f"{item['document_type_label']}: {item['document_number']}"
            )
            if item["notes"]:
                st.write(item["notes"])
            payload, filename, file_error = application_document_bytes(
                user["id"],
                item["id"],
            )
            if file_error:
                st.caption(file_error)
            elif payload:
                suffix = Path(filename).suffix.lower()
                if suffix in {".jpg", ".jpeg", ".png"}:
                    st.image(payload, caption=filename, width=280)
                st.download_button(
                    "Download card",
                    data=payload,
                    file_name=filename or f"application_{item['id']}",
                    key=f"ops_dl_agronomist_{item['id']}",
                )
            note = st.text_input(
                "Note",
                key=f"ops_agronomist_note_{item['id']}",
                placeholder="Required to deny",
            )
            accept, deny = st.columns(2)
            if accept.button(
                "Accept",
                key=f"ops_accept_agronomist_{item['id']}",
                type="primary",
            ):
                error = review_agronomist_application(
                    user["id"],
                    item["id"],
                    "approved",
                    note,
                )
                if error:
                    st.error(error)
                else:
                    st.session_state.ops_flash = (
                        f"{item['account_name']} is now a verified agronomist."
                    )
                    st.rerun()
            if deny.button(
                "Deny",
                key=f"ops_deny_agronomist_{item['id']}",
            ):
                error = review_agronomist_application(
                    user["id"],
                    item["id"],
                    "rejected",
                    note,
                )
                if error:
                    st.error(error)
                else:
                    st.session_state.ops_flash = "Application denied."
                    st.rerun()
    decided = approved + denied
    decided.sort(key=lambda item: item["reviewed_at"] or "", reverse=True)
    if decided:
        st.subheader("Decisions")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "When": _stamp(item["reviewed_at"] or item["submitted_at"]),
                        "Name": item["account_name"],
                        "Document": item["document_type_label"],
                        "Decision": (
                            "Accepted"
                            if item["status"] == "approved"
                            else "Denied"
                        ),
                        "Note": item["review_note"],
                    }
                    for item in decided[:12]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


def _operations(_user: dict, status: dict) -> None:
    _head(
        "Season control",
        f"The planner is on {status['latest_year']}. "
        f"Open or upload {status['next_year']} from Year intake.",
    )
    _flash()
    _kpis(
        [
            (
                "Latest season",
                str(status["latest_year"]),
                "Year the maps and estimates use",
            ),
            (
                "Next season",
                str(status["next_year"]),
                "The year Year intake can open",
            ),
            (
                "Storage",
                status["storage"],
                "Where this app is reading rows",
            ),
            (
                "Models",
                f"{status['models_ready']}/17",
                status["model_window"] or "Not trained",
            ),
        ]
    )
    years = pd.DataFrame(status["years"]).tail(8)
    if (
        int(years["Official yields"].sum()) == 0
        and int(years["Continued yields"].sum()) == 0
    ):
        years = years.drop(columns=["Official yields", "Continued yields"])
    years = years.rename(columns={"Role": "Status"})
    st.subheader("Recent seasons")
    st.dataframe(years, width="stretch", hide_index=True)


def _remember_results(summary: dict) -> None:
    _refresh_planner()
    st.session_state.ops_flash = summary["message"]
    st.session_state.ops_page = "Season results"
    st.session_state.ops_score_year = int(summary["latest_year"])
    st.session_state.ops_score_crop = "All crops"
    st.session_state.ops_force_score = True
    st.session_state.pop("ops_score_cache", None)


def _file_check(report) -> None:
    for error in report.file_errors:
        st.error(error)
    for warning in report.warnings:
        st.warning(warning)
    if report.row_errors:
        st.warning(
            "Some rows will be skipped.\n\n"
            + "\n".join(f"- {error}" for error in report.row_errors)
        )


def _intake(user: dict, status: dict) -> None:
    _head(
        "Year intake",
        "Harvests, weather, soil, and greenness each have their own upload. "
        "The planner reads the new numbers on the next calculation. "
        "Training stays off this page.",
    )
    _flash()
    latest = int(status["latest_year"])
    earliest = int(status["start_year"])
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Upload official results")
        st.markdown(
            "One row per district and crop. Leave a yield blank "
            "when that harvest is not in yet."
        )
        template_year = int(
            st.number_input(
                "Season in the template",
                min_value=earliest,
                max_value=latest + 8,
                value=min(latest + 1, latest + 8),
                step=1,
                key="ops_template_year",
            )
        )
        st.download_button(
            f"Download {template_year} template",
            data=template_csv(template_year),
            file_name=f"agrisight_{template_year}_yields.csv",
            mime="text/csv",
            width="stretch",
        )
        uploaded = st.file_uploader(
            "Official yield CSV",
            type=["csv"],
            key="ops_yield_upload",
        )

    with right:
        st.subheader("Open the next season")
        st.markdown(
            f"The registry ends in **{latest}**. "
            "This writes all 64 districts and 17 crops, "
            "then the planner uses the new year."
        )
        count = int(
            st.number_input(
                "Seasons to open",
                min_value=1,
                max_value=5,
                value=1,
                step=1,
                key="ops_open_count",
            )
        )
        opened_labels = ", ".join(
            str(latest + step) for step in range(1, count + 1)
        )
        st.caption(f"This will write {opened_labels}.")
        if st.button(
            f"Open {opened_labels}",
            type="primary",
            key="ops_open_season",
        ):
            try:
                summary = open_seasons(count, user["id"], user["name"])
            except ValueError as error:
                st.error(str(error))
            else:
                _remember_results(summary)
                st.rerun()

    if uploaded is not None:
        report = parse_upload(uploaded.getvalue())
        st.subheader("Harvest file check")
        _file_check(report)
        if not report.rows.empty:
            _yield_preview(latest, report, user)

    st.divider()
    _crop_file_panel(user, earliest, latest)
    st.divider()
    weather_col, soil_col = st.columns(2, gap="large")
    with weather_col:
        _weather_panel(user, earliest, latest)
    with soil_col:
        _soil_panel(user, earliest, latest)
    st.divider()
    green_col, _spare = st.columns(2, gap="large")
    with green_col:
        _greenness_panel(user, earliest, latest)


def _crop_file_panel(user: dict, earliest: int, latest: int) -> None:
    st.subheader("Update one crop")
    st.markdown(
        "Harvest only, for the crop you choose. A blank yield stays as it is. "
        "Weather, soil, and pH stay in the sections below, and greenness "
        "stays as stored."
    )
    choice_col, year_col = st.columns(2)
    labels = [spec["label"] for spec in CROP_CATALOG.values()]
    with choice_col:
        choice = st.selectbox("Crop", labels, key="ops_crop_file_crop")
    with year_col:
        year = int(
            st.number_input(
                "Year",
                min_value=earliest,
                max_value=latest + 8,
                value=latest,
                step=1,
                key="ops_crop_file_year",
            )
        )
    crop_id = _crop_choice_to_id(choice) or "boro_rice"
    payload, is_template = crop_year_csv(year, crop_id)
    kind = "template" if is_template else "file"
    st.download_button(
        f"Download {year} {choice} {kind}",
        data=payload,
        file_name=(
            f"agrisight_{year}_{crop_id}_template.csv"
            if is_template
            else f"{crop_id}_{year}.csv"
        ),
        mime="text/csv",
        width="stretch",
    )
    if is_template:
        st.caption("Blank harvests. This year is not stored yet.")
    uploaded = st.file_uploader(
        "Crop file CSV",
        type=["csv"],
        key="ops_crop_file_upload",
    )
    if uploaded is None:
        return
    report = parse_crop_rows(uploaded.getvalue())
    st.markdown("**Crop file check**")
    _file_check(report)
    if report.rows.empty:
        return
    preview = report.rows.copy()
    preview["Crop"] = preview["crop"].map(
        lambda item: CROP_CATALOG[item]["label"]
    )
    preview = preview.rename(
        columns={
            "year": "Year",
            "district": "District",
            "yield_ton_ha": "Yield (t/ha)",
        }
    )
    crops = sorted(preview["Crop"].unique())
    st.caption(
        f"{len(report.rows)} harvests will update {', '.join(crops)}. "
        f"{report.blank_rows} blank yields were left as stored."
    )
    st.dataframe(
        preview[["Year", "District", "Crop", "Yield (t/ha)"]].head(8),
        width="stretch",
        hide_index=True,
    )
    if st.button("Apply this crop file", type="primary", key="ops_apply_crop_file"):
        try:
            summary = apply_crop_rows(report.rows, user["id"], user["name"])
        except ValueError as error:
            st.error(str(error))
        else:
            _remember_results(summary)
            st.rerun()


def _yield_preview(latest: int, report, user: dict) -> None:

    ahead = [
        year
        for year in range(latest + 1, max(report.years) + 1)
    ]
    if ahead:
        labels = ", ".join(str(year) for year in ahead)
        st.info(
            f"This file also opens {labels} so the registry has no gap. "
            "Districts and crops missing from the file are continued."
        )
    preview = report.rows.copy()
    preview["Crop"] = preview["crop"].map(
        lambda item: CROP_CATALOG[item]["label"]
    )
    preview = preview.rename(
        columns={
            "year": "Year",
            "district": "District",
            "yield_ton_ha": "Yield (t/ha)",
        }
    )
    st.caption(
        f"{len(report.rows)} official yields · "
        f"{report.blank_rows} blank rows left for the system to continue · "
        f"{report.read_rows} rows read."
    )
    st.dataframe(
        preview[["Year", "District", "Crop", "Yield (t/ha)"]].head(12),
        width="stretch",
        hide_index=True,
    )
    if st.button(
        "Apply these results",
        type="primary",
        key="ops_apply_upload",
    ):
        try:
            summary = apply_upload(
                report.rows,
                user["id"],
                user["name"],
            )
        except ValueError as error:
            st.error(str(error))
        else:
            _remember_results(summary)
            st.rerun()


def _weather_panel(user: dict, earliest: int, latest: int) -> None:
    st.subheader("Update weather")
    st.markdown(
        "One row per district and year. Fill only the numbers you received. "
        "A blank cell keeps the weather already carried forward. "
        "Rainfall here is the seasonal total already stored with each crop, "
        "usually under a few hundred, in the same units as the crop file."
    )
    year = int(
        st.number_input(
            "Weather year",
            min_value=earliest,
            max_value=latest + 8,
            value=latest,
            step=1,
            key="ops_weather_year",
        )
    )
    st.download_button(
        f"Download {year} weather template",
        data=weather_template_csv(year),
        file_name=f"agrisight_{year}_weather.csv",
        mime="text/csv",
        width="stretch",
    )
    uploaded = st.file_uploader(
        "Weather CSV",
        type=["csv"],
        key="ops_weather_upload",
    )
    if uploaded is None:
        return
    report = parse_weather(uploaded.getvalue())
    st.markdown("**Weather file check**")
    _file_check(report)
    if report.rows.empty:
        return
    labels = {
        "rainfall_total": "Rainfall",
        "temp_max_mean": "Max °C",
        "temp_min_mean": "Min °C",
        "humidity_mean": "Humidity",
        "solar_radiation_mean": "Sunlight",
    }
    preview = report.rows.rename(
        columns={"year": "Year", "district": "District", **labels}
    )
    st.caption(
        f"{len(report.rows)} districts with a measurement · "
        f"{report.blank_rows} fully blank rows left carried forward."
    )
    st.dataframe(
        preview[["Year", "District", *labels.values()]].head(8),
        width="stretch",
        hide_index=True,
    )
    if st.button("Apply this weather", type="primary", key="ops_apply_weather"):
        try:
            summary = apply_weather(report.rows, user["id"], user["name"])
        except ValueError as error:
            st.error(str(error))
        else:
            _remember_results(summary)
            st.rerun()


def _soil_panel(user: dict, earliest: int, latest: int) -> None:
    st.subheader("Update soil and pH")
    st.markdown(
        "One row per district. The year is the year of the soil test. "
        "That district is updated from the test year onward. "
        "Older years stay as they were. A blank cell is left unchanged. "
        "CEC is included with pH, organic carbon, nitrogen, clay, and sand."
    )
    year = int(
        st.number_input(
            "Test year",
            min_value=earliest,
            max_value=latest + 8,
            value=latest,
            step=1,
            key="ops_soil_year",
        )
    )
    st.download_button(
        f"Download {year} soil template",
        data=soil_template_csv(year),
        file_name=f"agrisight_{year}_soil.csv",
        mime="text/csv",
        width="stretch",
    )
    uploaded = st.file_uploader(
        "Soil CSV",
        type=["csv"],
        key="ops_soil_upload",
    )
    if uploaded is None:
        return
    report = parse_soil(uploaded.getvalue())
    st.markdown("**Soil file check**")
    _file_check(report)
    if report.rows.empty:
        return
    labels = {
        "soil_ph": "pH",
        "soil_soc": "Organic carbon",
        "soil_nitrogen": "Nitrogen",
        "soil_clay": "Clay",
        "soil_sand": "Sand",
        "soil_cec": "CEC",
    }
    preview = report.rows.rename(
        columns={"year": "Year", "district": "District", **labels}
    )
    st.caption(
        f"{len(report.rows)} soil tests · "
        f"{report.blank_rows} fully blank rows left unchanged."
    )
    st.dataframe(
        preview[["Year", "District", *labels.values()]].head(8),
        width="stretch",
        hide_index=True,
    )
    if st.button("Apply these soil tests", type="primary", key="ops_apply_soil"):
        try:
            summary = apply_soil(report.rows, user["id"], user["name"])
        except ValueError as error:
            st.error(str(error))
        else:
            _remember_results(summary)
            st.rerun()


def _greenness_panel(user: dict, earliest: int, latest: int) -> None:
    st.subheader("Update greenness")
    st.markdown(
        "One row per district and year. Fill the satellite readings you have: "
        "the average, the peak, and how much it varied. "
        "A blank cell keeps the greenness already stored. "
        "Every crop in that district uses the new reading for that year."
    )
    year = int(
        st.number_input(
            "Greenness year",
            min_value=earliest,
            max_value=latest + 8,
            value=latest,
            step=1,
            key="ops_greenness_year",
        )
    )
    st.download_button(
        f"Download {year} greenness template",
        data=greenness_template_csv(year),
        file_name=f"agrisight_{year}_greenness.csv",
        mime="text/csv",
        width="stretch",
    )
    uploaded = st.file_uploader(
        "Greenness CSV",
        type=["csv"],
        key="ops_greenness_upload",
    )
    if uploaded is None:
        return
    report = parse_greenness(uploaded.getvalue())
    st.markdown("**Greenness file check**")
    _file_check(report)
    if report.rows.empty:
        return
    labels = {
        "ndvi_mean": "Average",
        "ndvi_peak": "Peak",
        "ndvi_std": "Variation",
    }
    preview = report.rows.rename(
        columns={"year": "Year", "district": "District", **labels}
    )
    st.caption(
        f"{len(report.rows)} districts with a reading · "
        f"{report.blank_rows} fully blank rows left as stored."
    )
    st.dataframe(
        preview[["Year", "District", *labels.values()]].head(8),
        width="stretch",
        hide_index=True,
    )
    if st.button("Apply this greenness", type="primary", key="ops_apply_greenness"):
        try:
            summary = apply_greenness(report.rows, user["id"], user["name"])
        except ValueError as error:
            st.error(str(error))
        else:
            _remember_results(summary)
            st.rerun()


def _crop_choice_to_id(choice: str) -> str | None:
    if choice == "All crops":
        return None
    for crop_id, spec in CROP_CATALOG.items():
        if spec["label"] == choice:
            return crop_id
    return None


def _results(status: dict) -> None:
    _head(
        "Season results",
        "Stored harvest next to the model. Rows marked Review need a check.",
    )
    _flash()
    years = [int(row["Year"]) for row in status["years"]]
    if st.session_state.get("ops_score_year") not in years:
        st.session_state.ops_score_year = years[-1]
    choices = ["All crops"] + [
        spec["label"] for spec in CROP_CATALOG.values()
    ]
    if st.session_state.get("ops_score_crop") not in choices:
        st.session_state.ops_score_crop = "All crops"

    year_col, crop_col, action_col = st.columns([1, 1.4, 1])
    with year_col:
        year = int(
            st.selectbox(
                "Season",
                years,
                key="ops_score_year",
            )
        )
    with crop_col:
        choice = st.selectbox(
            "Crop",
            choices,
            key="ops_score_crop",
        )
    with action_col:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        clicked = st.button(
            "Calculate this season",
            type="primary",
            key="ops_score_button",
        )

    crop_id = _crop_choice_to_id(choice)
    stamp = registry_stamp()
    cache_key = (year, choice, stamp)
    cached = st.session_state.get("ops_score_cache")
    force = st.session_state.pop("ops_force_score", False)
    if clicked or force or cached is None or cached.get("key") != cache_key:
        if not (clicked or force):
            st.caption(
                "Calculate this season to compare the stored harvest "
                "with the model."
            )
            return
        with st.spinner("Scoring the season…"):
            frame, warnings = score_season(year, crop_id)
        st.session_state.ops_score_cache = {
            "key": cache_key,
            "frame": frame,
            "warnings": warnings,
        }
    else:
        frame = cached["frame"]
        warnings = cached["warnings"]

    if "Source" in frame.columns:
        frame = frame.drop(columns=["Source"])

    for warning in warnings:
        st.warning(warning)
    if frame.empty:
        st.info("This season has no rows to score.")
        return

    review = int((frame["Check"] == "Review").sum())
    if "ops_only_review" not in st.session_state:
        st.session_state.ops_only_review = review > 0
    only_review = st.checkbox(
        "Show only rows to review",
        key="ops_only_review",
    )
    shown = frame[frame["Check"] == "Review"] if only_review else frame
    if review:
        st.caption(f"{review} rows differ enough to check.")
    st.download_button(
        f"Download {year} results",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=f"agrisight_{year}_season_results.csv",
        mime="text/csv",
    )
    if shown.empty:
        st.info("Nothing to review for this season.")
        return
    by_crop = (
        shown.groupby("Crop", as_index=False)
        .agg(
            Rows=("District", "count"),
            **{
                "Recorded (t/ha)": ("Recorded (t/ha)", "mean"),
                "Model (t/ha)": ("Model (t/ha)", "mean"),
                "To review": (
                    "Check",
                    lambda series: int((series == "Review").sum()),
                ),
            },
        )
    )
    by_crop["Recorded (t/ha)"] = by_crop["Recorded (t/ha)"].round(2)
    by_crop["Model (t/ha)"] = by_crop["Model (t/ha)"].round(2)
    st.subheader("By crop")
    st.dataframe(by_crop, width="stretch", hide_index=True)
    st.subheader("By district")
    st.dataframe(shown, width="stretch", hide_index=True)


def _registry(status: dict) -> None:
    _head(
        "Crop files",
        "These are the files behind the maps and estimates. "
        "Every crop should list the same districts and the same years.",
    )
    _flash()
    table = catalog_frame()
    expected_years = f"{status['start_year']}–{status['latest_year']}"
    district_count = int(status["districts"])
    short = table[
        (table["Districts"] < district_count)
        | (table["Years"] != expected_years)
    ]
    _kpis(
        [
            ("Crops", str(len(table)), "Each crop has its own file"),
            (
                "Districts",
                str(district_count),
                "Every crop should list all of them",
            ),
            (
                "Years",
                expected_years,
                "The span these files cover",
            ),
            (
                "Current year",
                str(status["latest_year"]),
                "The season the maps are using",
            ),
        ]
    )
    if short.empty:
        st.success(
            f"All {len(table)} crops cover {district_count} districts "
            f"from {status['start_year']} to {status['latest_year']}."
        )
    else:
        names = ", ".join(short["Crop"].astype(str))
        st.warning(
            f"{names} do not match the others. "
            "A shorter year list means that crop is behind."
        )
    show = table.rename(
        columns={
            "Season": "When it grows",
            "Years": "Years in the file",
            "Mean yield (t/ha)": "Average yield (t/ha)",
        }
    )[
        [
            "Crop",
            "Districts",
            "Years in the file",
            "Average yield (t/ha)",
            "When it grows",
        ]
    ]
    st.caption(
        "Average yield is the mean of every year in the file, "
        "not the latest season by itself. "
        "One line in a downloaded file is one district in one year."
    )
    st.dataframe(show, width="stretch", hide_index=True)
    st.subheader("Save a copy")
    st.markdown(
        "Download the file the planner is reading. "
        "Choose one crop, or save every crop together."
    )
    labels = [spec["label"] for spec in CROP_CATALOG.values()]
    choice = st.selectbox("Crop", labels, key="ops_registry_crop")
    crop_id = _crop_choice_to_id(choice)
    download_col, zip_col = st.columns(2)
    with download_col:
        st.download_button(
            f"Download {choice}",
            data=crop_file_bytes(crop_id),
            file_name=f"{crop_id}.csv",
            mime="text/csv",
            width="stretch",
        )
    with zip_col:
        st.download_button(
            "Download every crop",
            data=datasets_zip_bytes(),
            file_name="agrisight_crop_datasets.zip",
            mime="application/zip",
            width="stretch",
        )


REVIEW_STATUS_LABELS = {
    "submitted": "Submitted · awaiting assignment",
    "assigned": "Assigned · awaiting agronomist",
    "in_review": "In review",
    "reviewed": "Reviewed by agronomist",
    "closed": "Closed without review",
}


def _review_case(user: dict, case: dict, reviewers: list[dict]) -> None:
    is_admin = user["role"] == "admin"
    eligible_reviewers = [
        person
        for person in reviewers
        if person["id"] != case["requester_user_id"]
    ]
    title = (
        f"Case #{case['id']} · {case['crop_name']} · {case['district']} · "
        f"{REVIEW_STATUS_LABELS.get(case['status'], case['status'])}"
    )
    with st.expander(title):
        st.caption(
            f"Submitted {case['created_at']} · "
            f"{case['requester_name']} · {case['contact_email']}"
        )
        st.markdown("**Question**")
        st.caption(
            f"Topic: {REVIEW_CATEGORIES.get(case['category'], case['category'])}"
        )
        st.write(case["question"])
        st.markdown("**Prediction context**")
        st.write(
            f"Estimate: {case['predicted_yield']:.2f} t/ha · "
            f"District usual: {case['usual_yield']:.2f} t/ha"
        )
        st.json(case["details"])

        if is_admin:
            if case["status"] not in ("reviewed", "closed"):
                if eligible_reviewers:
                    reviewer_ids = [
                        person["id"] for person in eligible_reviewers
                    ]
                    reviewer_labels = {
                        person["id"]: (
                            f"{person['name']} · "
                            f"{person['organization'] or 'No organization'} · "
                            f"{person['qualifications']}"
                        )
                        for person in eligible_reviewers
                    }
                    current_index = (
                        reviewer_ids.index(case["assigned_to"])
                        if case["assigned_to"] in reviewer_ids
                        else 0
                    )
                    with st.form(f"assign_review_{case['id']}"):
                        reviewer_id = st.selectbox(
                            "Assign to verified agronomist",
                            reviewer_ids,
                            index=current_index,
                            format_func=lambda item: reviewer_labels[item],
                            key=f"reviewer_for_{case['id']}",
                        )
                        assigned = st.form_submit_button(
                            "Assign or reassign case"
                        )
                    if assigned:
                        error = assign_review_request(
                            user["id"],
                            case["id"],
                            reviewer_id,
                        )
                        if error:
                            st.error(error)
                        else:
                            st.rerun()
                else:
                    st.warning(
                        "No admin-verified agronomist is available. Verify "
                        "a qualified reviewer from Applications before assigning cases."
                    )
                if st.button(
                    "Close without review",
                    key=f"close_review_{case['id']}",
                ):
                    error = close_review_request(user["id"], case["id"])
                    if error:
                        st.error(error)
                    else:
                        st.rerun()
        elif case["status"] == "assigned":
            if st.button(
                "Start review",
                key=f"start_review_{case['id']}",
                type="primary",
            ):
                error = start_review_request(user["id"], case["id"])
                if error:
                    st.error(error)
                else:
                    st.rerun()
        elif case["status"] == "in_review":
            with st.form(f"complete_review_{case['id']}"):
                response = st.text_area(
                    "Agronomist review and recommended next steps",
                    max_chars=2000,
                    key=f"review_response_{case['id']}",
                    help=(
                        "Explain the evidence, uncertainties, and any "
                        "on-site checks needed."
                    ),
                )
                completed = st.form_submit_button(
                    "Submit review to requester",
                    type="primary",
                )
            if completed:
                error = complete_review_request(
                    user["id"],
                    case["id"],
                    response,
                )
                if error:
                    st.error(error)
                else:
                    st.rerun()

        if case["response"]:
            st.markdown("**Agronomist response**")
            st.write(case["response"])

        st.markdown("**Audit history**")
        for event in list_review_events(user["id"], case["id"]):
            change = (
                f" · {event['from_status']} → {event['to_status']}"
                if event["to_status"]
                else ""
            )
            st.caption(
                f"{event['created_at']} · {event['action'].replace('_', ' ').title()}"
                f"{change} · {event['actor_name']}"
            )


def _status_count(requests: list[dict], status: str) -> int:
    return sum(1 for case in requests if case["status"] == status)


def _review_inbox(user: dict) -> None:
    is_admin = user["role"] == "admin"
    _head("Expert review" if is_admin else "Assigned cases")
    if is_admin:
        reviewers = list_verified_agronomists(user["id"])
    else:
        reviewers = []
    requests = list_review_requests(user["id"])
    submitted = _status_count(requests, "submitted")
    assigned = _status_count(requests, "assigned")
    in_review = _status_count(requests, "in_review")
    reviewed = _status_count(requests, "reviewed")
    closed = _status_count(requests, "closed")
    if is_admin:
        _kpis(
            [
                ("Open", str(submitted + assigned + in_review), "Still in the queue"),
                ("Unassigned", str(submitted), "Waiting for a reviewer"),
                ("With reviewer", str(assigned + in_review), "Assigned or in review"),
                ("Reviewed", str(reviewed), "Reply sent"),
                ("Closed", str(closed), "Closed without a reply"),
                ("Reviewers", str(len(reviewers)), "Verified agronomists"),
            ],
            columns=3,
        )
        if reviewers:
            cards = "".join(
                "<span>"
                f"{html.escape(person['name'])}"
                f"<em>{html.escape(person['organization'] or 'Independent')}</em>"
                "</span>"
                for person in reviewers
            )
            st.markdown(
                f'<div class="reviewer-row">{cards}</div>',
                unsafe_allow_html=True,
            )
    else:
        _kpis(
            [
                ("Assigned", str(assigned), "Not started"),
                ("In review", str(in_review), "Reply in progress"),
                ("Reviewed", str(reviewed), "Sent to the requester"),
            ]
        )
    if not requests:
        return
    for case in requests:
        _review_case(user, case, reviewers)


def render_admin_app(user: dict) -> None:
    """Render the admin console or restricted agronomist review desk."""

    if user["role"] != "admin" and not (
        user["role"] == "agronomist"
        and user["expert_verified"]
        and user["active"]
    ):
        st.error("This account is not authorized for the operations console.")
        return
    _shell()
    page = _sidebar(user)
    if page == "Expert review":
        _review_inbox(user)
        return
    if user["role"] != "admin":
        st.error("Agronomist accounts can only access assigned review cases.")
        return
    if page == "Applications":
        _applications(user)
        return
    status = season_status()
    if page == "Operations":
        _operations(user, status)
    elif page == "Year intake":
        _intake(user, status)
    elif page == "Season results":
        _results(status)
    elif page == "Crop files":
        _registry(status)
    else:
        _operations(user, status)
