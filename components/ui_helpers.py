import os
import re
import streamlit as st

# Project root (parent of components/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def inject_custom_css():
    """Reads style.css from project root and injects it into the Streamlit session."""
    css_path = os.path.join(_PROJECT_ROOT, "style.css")
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception:
        pass


def risk_badge(risk_level: str) -> str:
    """Returns the HTML string for a colored status badge."""
    level_lower = risk_level.lower()
    if level_lower == "high":
        return '<span class="badge badge-high">High Risk</span>'
    if level_lower == "medium":
        return '<span class="badge badge-medium">Medium Risk</span>'
    return '<span class="badge badge-low">Low Risk</span>'


def _normalize_trend(trend_html: str) -> str:
    """Wraps raw trend HTML in a styled pill if not already wrapped."""
    if not trend_html:
        return ""
    if "kpi-trend" in trend_html:
        return trend_html
    color_class = "kpi-trend--neutral"
    if any(c in trend_html for c in ("#EF4444", "#F59E0B", "Critical", "Alert", "-")):
        color_class = "kpi-trend--down" if "Critical" in trend_html or "#EF4444" in trend_html else "kpi-trend--warn"
    elif any(c in trend_html for c in ("#22C55E", "Stable", "Active", "+")):
        color_class = "kpi-trend--up"
    # Strip outer span tags for clean text
    text = re.sub(r"<[^>]+>", "", trend_html).strip()
    return f'<span class="kpi-trend {color_class}">{text}</span>'


def glass_card(
    title: str,
    value: str,
    subtitle: str = None,
    trend_html: str = "",
    icon: str = "",
    accent: str = "primary",
) -> str:
    """Renders a premium KPI card."""
    sub_text = f'<div class="kpi-subtitle">{subtitle}</div>' if subtitle else ""
    icon_html = f'<div class="kpi-card__icon">{icon}</div>' if icon else ""
    trend = _normalize_trend(trend_html)

    return f"""
    <div class="kpi-card animate-fade-in">
        <div class="kpi-card__accent kpi-card__accent--{accent}"></div>
        <div class="kpi-card__header">
            <div>
                <div class="kpi-title">{title}</div>
            </div>
            {icon_html}
        </div>
        <div style="display:flex; align-items:baseline; justify-content:space-between; gap:8px; flex-wrap:wrap;">
            <div class="kpi-value">{value}</div>
            <div>{trend}</div>
        </div>
        {sub_text}
    </div>
    """


def render_section_header(title: str, subtitle: str = None):
    """Renders a section header safely without letting whitespace break rendering."""
    subtitle_html = f'<p class="section-header__subtitle">{subtitle}</p>' if subtitle else ""
    
    # Combined into a single line HTML string to avoid Streamlit markdown parsing bug
    html_content = f'<div class="section-header animate-fade-in"><h2 class="section-header__title">{title}</h2>{subtitle_html}<div class="section-header__accent"></div></div>'
    
    st.markdown(html_content, unsafe_allow_html=True)


def render_page_hero(title: str, subtitle: str = None, eyebrow: str = "ExplainChurn AI"):
    """Optional page-level hero banner for the active view."""
    subtitle_html = f'<p class="page-hero__desc">{subtitle}</p>' if subtitle else ""
    html_content = f'<div class="page-hero animate-fade-in"><div class="page-hero__eyebrow">{eyebrow}</div><h1 class="page-hero__title">{title}</h1>{subtitle_html}</div>'
    st.markdown(html_content, unsafe_allow_html=True)


def render_sidebar_brand():
    """Premium sidebar brand block."""
    st.sidebar.markdown(
        '<div class="sidebar-brand animate-fade-in"><div class="sidebar-brand__logo">🔮</div><h2 class="sidebar-brand__title gradient-text">EXPLAINCHURN</h2><p class="sidebar-brand__tagline">Enterprise Churn Intelligence</p></div>',
        unsafe_allow_html=True,
    )


def render_sidebar_nav_label():
    """Navigation section label in sidebar."""
    st.sidebar.markdown(
        '<div class="sidebar-nav-label">Main Navigation</div>',
        unsafe_allow_html=True,
    )


def render_sidebar_telemetry(
    algorithm: str = "XGBoost v2.1.4",
    engine_status: str = "Active",
    platform_health: str = "100%",
):
    """Model telemetry panel pinned to sidebar bottom area."""
    st.sidebar.markdown(
        f'<div class="sidebar-telemetry animate-fade-in"><div class="sidebar-telemetry__title">Model Telemetry</div><div class="sidebar-telemetry__row"><span class="sidebar-telemetry__label">Algorithm</span><span class="sidebar-telemetry__value">{algorithm}</span></div><div class="sidebar-telemetry__row"><span class="sidebar-telemetry__label">Engine</span><span class="status-dot">{engine_status}</span></div><div class="sidebar-telemetry__row"><span class="sidebar-telemetry__label">Health</span><span class="sidebar-telemetry__value">{platform_health}</span></div></div>',
        unsafe_allow_html=True,
    )


def render_footer():
    """Renders a luxury styled platform footer."""
    st.markdown(
        '<div class="app-footer animate-fade-in"><div class="app-footer__links"><a href="https://github.com" target="_blank" class="app-footer__link">GitHub</a><a href="https://linkedin.com" target="_blank" class="app-footer__link">LinkedIn</a><a href="mailto:support@explainchurn.ai" class="app-footer__link">Email Support</a><a href="#" class="app-footer__link">Documentation</a></div><div class="app-footer__meta">ExplainChurn AI · Enterprise Churn Prediction Platform<span class="app-footer__badge">v1.2.0 Production</span></div></div>',
        unsafe_allow_html=True,
    )


def skeleton_loader(num_rows: int = 2, variant: str = "card"):
    """Displays skeleton loading placeholders."""
    if variant == "kpi":
        st.markdown(
            '<div class="skeleton-kpi-grid">' + "".join('<div class="skeleton-kpi"></div>' for _ in range(4)) + "</div>",
            unsafe_allow_html=True,
        )
        return

    if variant == "chart":
        st.markdown('<div class="skeleton-chart"></div>', unsafe_allow_html=True)
        return

    cols = st.columns(3)
    for _ in range(num_rows):
        for col in cols:
            col.markdown('<div class="skeleton-loader"></div>', unsafe_allow_html=True)


def render_loading_state(message: str = "Loading data…"):
    """Full-width loading skeleton with message."""
    skeleton_loader(variant="kpi")
    st.markdown(
        f'<div class="loading-overlay-text">{message}</div>',
        unsafe_allow_html=True,
    )


def render_notification(message: str, type_: str = "info"):
    """Displays a toast notification style alert."""
    type_map = {
        "info": ("ℹ️", "notification--info"),
        "alert": ("🚨", "notification--alert"),
        "success": ("✅", "notification--success"),
        "warning": ("⚠️", "notification--warning"),
    }
    icon, css_class = type_map.get(type_, type_map["info"])

    st.markdown(
        f'<div class="notification {css_class}"><div class="notification__icon">{icon}</div><div class="notification__text">{message}</div></div>',
        unsafe_allow_html=True,
    )


def open_panel(title: str = None, label: str = None) -> str:
    """Returns opening HTML for a styled panel card."""
    label_html = f'<div class="panel-card__label">{label}</div>' if label else ""
    title_html = f'<div class="panel-card__title">{title}</div>' if title else ""
    return f'<div class="panel-card animate-fade-in">{label_html}{title_html}'


def close_panel() -> str:
    """Returns closing HTML for a panel card."""
    return "</div>"