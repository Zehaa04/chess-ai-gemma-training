import os
import json
import chess
import chess.pgn
from pathlib import Path
import re

BASE_DIR = Path("chess_dataset")
PGN_DIR = BASE_DIR
IMG_DIR = BASE_DIR / "screenshots"
OUT_FILE = BASE_DIR / "chess_with_fen.jsonl"

def sanitize(s):
    s = s.replace(" ", "_")
    s = re.sub(r"[^\w\-\.\[\]]", "", s)
    return s

def make_record(img_path, question, answer, fen):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            },
            {
                "role": "metadata",
                "content": [
                    {"type": "text", "text": fen}
                ]
            }
        ]
    }

with open(OUT_FILE, "w", encoding="utf-8") as out_f:
    count_games = 0

    for pgn_path in PGN_DIR.glob("game*.pgn"):
        count_games += 1
        print(f"Processing {pgn_path.name}")

        game_id = pgn_path.stem  # "game12"

        with open(pgn_path, "r", encoding="utf-8") as f:
            game = chess.pgn.read_game(f)

        board = game.board()
        ply = 0

        for move in game.mainline_moves():
            san = board.san(move)
            ply += 1
            board.push(move)

            fen = board.fen()

            san_safe = sanitize(san)
            img_name = f"{game_id}_{san_safe}_ply{ply}.png"
            img_path = IMG_DIR / img_name

            if not img_path.exists():
                print(f"WARNING: Missing image {img_path}")
                continue

            
            rec1 = make_record(
                img_path,
                "What move was played?",
                san,
                fen
            )
            out_f.write(json.dumps(rec1) + "\n")

            piece_symbol = board.piece_at(move.to_square).symbol().lower()
            PIECE_MAP = {
                "p": "pawn",
                "n": "knight",
                "b": "bishop",
                "r": "rook",
                "q": "queen",
                "k": "king"
            }
            rec2 = make_record(
                img_path,
                "Which piece moved?",
                PIECE_MAP[piece_symbol],
                fen
            )
            out_f.write(json.dumps(rec2) + "\n")

rec3 = make_record(
    img_path,
    "Was it a capture?",
    "yes" if board.is_capture(move) else "no",
    fen
)
out_f.write(json.dumps(rec3) + "\n")


print(f"DONE. Parsed {count_games} games. Output saved to {OUT_FILE}")
