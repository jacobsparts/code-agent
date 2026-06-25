import os
import sys


def _fit(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 1:
        return "…"
    return text[:width - 1] + "…"


def _get_term_size():
    try:
        sz = os.get_terminal_size()
        return sz.columns, sz.lines
    except OSError:
        return 100, 30


def _render(items, selected, scroll_offset, term_width, term_height, current_model):
    out = ['\x1b[?25l', '\x1b[2J', '\x1b[H']
    title = ' /model '
    out.append(f'\x1b[7m{title:<{term_width}}\x1b[0m\n')
    subtitle = '  ↑/↓ navigate | Enter select model | Esc cancel'
    out.append(f'\x1b[2m{_fit(subtitle, term_width):<{term_width}}\x1b[0m\n\n')

    header_lines = 3
    footer_lines = 2
    available = term_height - header_lines - footer_lines
    items_visible = max(1, available)
    visible = items[scroll_offset:scroll_offset + items_visible]

    for i, item in enumerate(visible):
        idx = scroll_offset + i
        is_sel = idx == selected
        is_current = item["full_name"] == current_model
        marker = '>' if is_sel else ' '
        checkbox = '[x]' if is_current else '[ ]'
        style = '\x1b[7m' if is_sel else ''
        end = '\x1b[0m' if style else ''
        aliases = item.get("aliases") or []
        alias_text = f" ({', '.join(aliases)})" if aliases else ""
        line = f' {marker} {checkbox} {item["full_name"]}{alias_text}'
        out.append(f'{style}{line[:term_width]:<{term_width}}{end}\n')

    selected_item = items[selected]
    footer = f'  {len(items)} models  •  current: {current_model}  •  selected: {selected_item["full_name"]}'
    out.append(f'\x1b[2m{footer[:term_width]:<{term_width}}\x1b[0m')
    return ''.join(out)


def _render_empty(term_width, term_height):
    out = ['\x1b[?25l', '\x1b[2J', '\x1b[H']
    title = ' /model '
    out.append(f'\x1b[7m{title:<{term_width}}\x1b[0m\n')
    subtitle = '  Esc cancel'
    out.append(f'\x1b[2m{_fit(subtitle, term_width):<{term_width}}\x1b[0m\n\n')
    out.append('  No models found.\n')
    return ''.join(out)


def select_model_ui(altmode, models: list[dict], current_model: str) -> str | None:
    from .prompt import RawMode

    items = [dict(item) for item in models]
    selected = 0
    for idx, item in enumerate(items):
        if item["full_name"] == current_model:
            selected = idx
            break
    scroll_offset = 0

    session = altmode.session()
    session.enter()
    try:
        with RawMode():
            while True:
                term_width, term_height = _get_term_size()
                items_visible = max(1, term_height - 5)
                if items:
                    if selected < scroll_offset:
                        scroll_offset = selected
                    if selected >= scroll_offset + items_visible:
                        scroll_offset = selected - items_visible + 1
                    sys.stdout.write(_render(items, selected, scroll_offset, term_width, term_height, current_model))
                else:
                    sys.stdout.write(_render_empty(term_width, term_height))
                sys.stdout.flush()
                k = os.read(sys.stdin.fileno(), 4096)
                if not k:
                    continue
                c = k[0]
                if c in (3, 27) and len(k) == 1:
                    return None
                if not items:
                    continue
                if c in (10, 13):
                    return items[selected]["full_name"]
                if c == 27 and len(k) >= 3 and k[1] == 91:
                    if k[2] == 65:
                        selected = max(0, selected - 1)
                    elif k[2] == 66:
                        selected = min(len(items) - 1, selected + 1)
    finally:
        sys.stdout.write('\x1b[?25h')
        sys.stdout.flush()
        session.exit()
