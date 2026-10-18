"""Small, streaming terminal presentation; never changes model input or memory."""
import os
import sys

RESET = "\x1b[0m"
BODY = "\x1b[36m"
HEADING = "\x1b[1;34m"
BOLD = "\x1b[1;36m"
CODE = "\x1b[33m"


def supports_color(stream=None):
    stream = sys.stdout if stream is None else stream
    return stream.isatty() and os.environ.get("TERM", "dumb") != "dumb" and "NO_COLOR" not in os.environ


def shell_prompt(identity, directory):
    """Arguments have already been sanitized; add only trusted ANSI sequences.

    Readline markers make color escapes zero-width for cursor motion/wrapping.
    Match the user's Bash colors without evaluating PS1 or sourcing startup code.
    """
    if not sys.stdin.isatty() or not supports_color():
        return f"{identity}:{directory}$ "
    green = "\001\x1b[01;32m\002"
    blue = "\001\x1b[01;34m\002"
    reset = "\001" + RESET + "\002"
    return f"{green}{identity}{reset}:{blue}{directory}{reset}$ "


class AnswerStream:
    """Render common Markdown incrementally, keeping at most a delimiter suffix.

    No full-answer/line buffering, Markdown execution, hyperlinks or dependencies.
    Redirected output remains the original Markdown. Input is already sanitized
    by the agent; the only added terminal controls are this module's fixed colors.
    """
    def __init__(self, stream=None):
        self.stream = sys.stdout if stream is None else stream
        self.colored = supports_color(self.stream)
        self.pending = ""
        self.started = False
        self.bold = self.inline = self.block = self.heading = False
        self.line_start = True

    def style(self):
        return CODE if self.block or self.inline else HEADING if self.heading else BOLD if self.bold else BODY

    def write(self, text):
        if not text:
            return
        if not self.colored:
            self.stream.write(text)
            self.stream.flush()
            return
        self.pending += text
        self._render(False)

    def _render(self, final):
        text, self.pending = self.pending, ""
        output = []
        if text and not self.started:
            output.append(BODY)
            self.started = True
        i = 0
        while i < len(text):
            rest = text[i:i + 7]  # Bounded lookahead keeps rendering linear.
            char = text[i]
            if char == "\\" and not self.block:
                if not final and rest == "\\":
                    self.pending = rest
                    break
                if len(rest) > 1 and rest[1] in "\\`*#":
                    output.append(rest[1])
                    self.line_start = False
                    i += 2
                    continue
            if char == "`":
                if not final and len(rest) < 3 and not rest.strip("`"):
                    self.pending = rest
                    break
                if self.line_start and rest.startswith("```"):
                    self.block = not self.block
                    self.inline = False
                    output.append(RESET + self.style())
                    i += 3
                    continue
                if not self.block:
                    self.inline = not self.inline
                    output.append(RESET + self.style())
                    self.line_start = False
                    i += 1
                    continue
            if not self.block and not self.inline:
                if char == "*":
                    if not final and rest == "*":
                        self.pending = rest
                        break
                    if rest.startswith("**"):
                        self.bold = not self.bold
                        output.append(RESET + self.style())
                        self.line_start = False
                        i += 2
                        continue
                if self.line_start and char == "#":
                    count = len(rest) - len(rest.lstrip("#"))
                    if not final and count == len(rest) and count <= 6:
                        self.pending = rest
                        break
                    if 1 <= count <= 6 and rest[count:count + 1] == " ":
                        self.heading = True
                        output.append(RESET + HEADING)
                        self.line_start = False
                        i += count + 1
                        continue
            output.append(char)
            if char == "\n":
                self.line_start = True
                if self.heading:
                    self.heading = False
                    output.append(RESET + self.style())
            elif not char.isspace():
                self.line_start = False
            i += 1
        if output:
            self.stream.write("".join(output))
            self.stream.flush()

    def finish(self):
        if self.colored:
            self._render(True)
            if self.started:
                self.stream.write(RESET)
                self.stream.flush()
                self.started = False
