import os
import re
import time
import chess
import chess.pgn
import chess.svg
import chess.engine
import cairosvg
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
matches = list((BASE_DIR / "stockfish").glob("*.exe"))
if not matches:
    raise FileNotFoundError("No Stockfish executable found in ./stockfish/")
STOCKFISH_PATH = matches[0]
OUT_DIR = Path("chess_dataset")

GAMES = 100
MAX_MOVES = 300
THINK_TIME = 0.05

OUT_DIR.mkdir(parents=True, exist_ok=True)
screens_dir = OUT_DIR / "screenshots"
screens_dir.mkdir(parents=True, exist_ok=True)
pgn_master_path = OUT_DIR / "all_games.pgn"

engine_white = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
engine_black = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

def sanitize(s):
    s = s.replace(" ", "_")
    s = re.sub(r"[^\w\-\.\[\]]", "", s)
    return s

master_file = pgn_master_path.open("w", encoding="utf-8")
for g in range(1, GAMES + 1):
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    moves_san = []
    ply = 0
    while not board.is_game_over() and ply < MAX_MOVES:
        engine = engine_white if board.turn == chess.WHITE else engine_black
        engine_name = "White" if board.turn == chess.WHITE else "Black"
        try:
            result = engine.play(board, chess.engine.Limit(time=THINK_TIME))
        except Exception:
            break
        move = result.move
        if move is None:
            break
        san = board.san(move)
        ply += 1
        board.push(move)
        moves_san.append(san)
        svg = chess.svg.board(board=board, size=600)
        fname = f"game{g}_{sanitize(san)}_ply{ply}.png"
        path = screens_dir / fname
        try:
            cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(path))
        except Exception:
            pass
        node = node.add_variation(move)
    game.headers["Event"] = f"EngineMatch_game{g}"
    game.headers["White"] = "White"
    game.headers["Black"] = "Black"
    game.headers["Result"] = board.result() if board.is_game_over() else "*"
    with (OUT_DIR / f"game{g}.pgn").open("w", encoding="utf-8") as f:
        exporter = chess.pgn.FileExporter(f)
        game.accept(exporter)
    master_file.write(game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False)) + "\n\n")
    time.sleep(0.01)
master_file.close()
engine_white.quit()
engine_black.quit()
