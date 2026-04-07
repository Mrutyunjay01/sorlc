import tkinter as tk
import chess  # type: ignore[reportMissingImports]


UNICODE_PIECES = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
}


class ChessTkUI:
    def __init__(self, title: str = "Chess Live View", cell_px: int = 64):
        self.cell_px = cell_px
        self.board_px = 8 * cell_px

        self.root = tk.Tk()
        self.root.title(title)

        self.canvas = tk.Canvas(self.root, width=self.board_px, height=self.board_px, highlightthickness=0)
        self.canvas.pack()

        self.status_var = tk.StringVar(value="Ready")
        self.status = tk.Label(self.root, textvariable=self.status_var, anchor="w")
        self.status.pack(fill="x")

        self._draw_board_base()
        self.root.update_idletasks()
        self.root.update()

    def _draw_board_base(self) -> None:
        # Strong but not harsh
        self._light = "#EDEDED"
        self._dark  = "#3A3A3A"

        c = self.cell_px
        for rank in range(8):
            for file in range(8):
                x0, y0 = file * c, rank * c
                x1, y1 = x0 + c, y0 + c

                color = self._light if (rank + file) % 2 == 0 else self._dark

                self.canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill=color,
                    outline=color
                )

    def render(self, fen: str, status_text: str = "", last_move: str | None = None) -> None:
        self.canvas.delete("piece")
        board = chess.Board(fen=fen)
        c = self.cell_px

        for square, piece in board.piece_map().items():
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)

            x = file_idx * c + c / 2
            y = (7 - rank_idx) * c + c / 2

            symbol = UNICODE_PIECES[piece.symbol()]

            font_size = int(c * 0.7)

            # --- TRUE piece colors (fixed) ---
            if piece.color:  # white
                fill = "#F8F8F8"
                outline = "#111111"
            else:  # black
                fill = "#111111"
                outline = "#F0F0F0"

            # --- subtle drop shadow (depth) ---
            self.canvas.create_text(
                x + 2, y + 2,
                text=symbol,
                font=("Segoe UI Symbol", font_size),
                fill="#000000",
                tags="piece"
            )

            # --- stroke / outline ---
            for dx, dy in [
                (-1,0),(1,0),(0,-1),(0,1),
                (-1,-1),(1,1),(-1,1),(1,-1)
            ]:
                self.canvas.create_text(
                    x + dx, y + dy,
                    text=symbol,
                    font=("Segoe UI Symbol", font_size),
                    fill=outline,
                    tags="piece"
                )

            # --- main glyph ---
            self.canvas.create_text(
                x, y,
                text=symbol,
                font=("Segoe UI Symbol", font_size),
                fill=fill,
                tags="piece"
            )

        suffix = f" | Last move: {last_move}" if last_move else ""
        self.status_var.set(f"{status_text}{suffix}".strip())

        self.root.update_idletasks()
        self.root.update()

    def prompt_move(self, observation) -> str:
        legal_moves = list(getattr(observation, "legal_moves", []))
        if not legal_moves:
            raise ValueError("no legal moves available for manual UI input")

        selected: list[str] = []
        chosen_move: dict[str, str | None] = {"move": None}
        status_before = self.status_var.get()
        self.status_var.set("Your move: click source square, then destination square.")

        def choose_from_candidates(candidates: list[str]) -> str:
            if len(candidates) == 1:
                return candidates[0]
            for promotion_piece in ("q", "r", "b", "n"):
                promoted = [m for m in candidates if len(m) == 5 and m[-1] == promotion_piece]
                if promoted:
                    return promoted[0]
            return candidates[0]

        def on_click(event):
            if event.x < 0 or event.y < 0 or event.x >= self.board_px or event.y >= self.board_px:
                return

            file_idx = int(event.x // self.cell_px)
            rank_idx = 7 - int(event.y // self.cell_px)
            square_name = chess.square_name(chess.square(file_idx, rank_idx))
            selected.append(square_name)

            if len(selected) == 1:
                self.status_var.set(f"Selected {selected[0]}. Now click destination square.")
                return

            if len(selected) == 2:
                move_prefix = selected[0] + selected[1]
                candidates = [m for m in legal_moves if m.startswith(move_prefix)]
                if not candidates:
                    self.status_var.set("Illegal move pair. Select source square again.")
                    selected.clear()
                    return

                chosen_move["move"] = choose_from_candidates(candidates)
                self.canvas.unbind("<Button-1>")

        self.canvas.bind("<Button-1>", on_click)
        while chosen_move["move"] is None:
            self.root.update_idletasks()
            self.root.update()

        self.status_var.set(status_before)
        return chosen_move["move"]  # type: ignore[return-value]

    def close(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass
