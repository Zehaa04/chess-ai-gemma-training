import os
import json
import base64
import chess.pgn
import chess.svg
from cairosvg import svg2png

PGN_DIR = "./chess_dataset"            
OUTPUT_JSONL = "chess_dataset.jsonl"

def png_base64(board):
    svg_data = chess.svg.board(board=board, size=350)
    png_bytes = svg2png(bytestring=svg_data)
    return base64.b64encode(png_bytes).decode("utf-8")

def make_record(user_q, img_b64, answer, fen):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_b64},
                    {"type": "text",  "text": user_q}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}]
            },
            {
                "role": "metadata",
                "content": [{"type": "text", "text": fen}]
            }
        ]
    }

with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:

    files = [f for f in os.listdir(PGN_DIR) if f.lower().endswith(".pgn")]
    files.sort()

    print("Found", len(files), "PGN files")

    for fname in files:
        pgn_path = os.path.join(PGN_DIR, fname)
        print("Processing", pgn_path)

        with open(pgn_path, "r", encoding="utf-8") as f:
            game = chess.pgn.read_game(f)

            if game is None:
                print("WARNING: Skipping empty or invalid file:", fname)
                continue

            board = game.board()

            for move in game.mainline_moves():

              san = board.san(move)
              moved_piece = board.piece_at(move.from_square)
              piece_name = moved_piece.piece_type if moved_piece else "?"

              board.push(move)

              fen = board.fen()
              img_b64 = png_base64(board)

              out_f.write(json.dumps(
                  make_record("What move was played?", img_b64, san, fen)
              ) + "\n")

              out_f.write(json.dumps(
                  make_record("Which piece moved?", img_b64, piece_name, fen)
              ) + "\n")

print("DONE. Wrote:", OUTPUT_JSONL)
