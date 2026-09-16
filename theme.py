# =========================================================
# LIGHT THEME
# =========================================================

LIGHT_THEME = {
    "bg": "#F5F5F5",
    "fg": "#000000",
    "button_bg": "#E0E0E0",
    "button_fg": "#000000"
}


# =========================================================
# DARK THEME
# =========================================================

DARK_THEME = {
    "bg": "#1E1E1E",
    "fg": "#FFFFFF",
    "button_bg": "#404040",
    "button_fg": "#FFFFFF"
}


# =========================================================
# CURRENT THEME
# =========================================================

current_theme = LIGHT_THEME


def get_theme():
    return current_theme


def toggle_theme():

    global current_theme

    if current_theme == LIGHT_THEME:
        current_theme = DARK_THEME
    else:
        current_theme = LIGHT_THEME

    return current_theme
