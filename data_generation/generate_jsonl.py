import os
import json
import chess
import chess.pgn
import random

# --- Configuration ---
PGN_FOLDER = "pgns"
IMG_FOLDER = "screenshots_200games"
OUTPUT_FOLDER = "output"
JSONL_VISUAL_BOARD = os.path.join(OUTPUT_FOLDER, "stage1_visual_board.jsonl")
JSONL_VISUAL_RCD = os.path.join(OUTPUT_FOLDER, "stage1_visual_rcd.jsonl")  # Row, Column, Diagonal
JSONL_STRATEGY = os.path.join(OUTPUT_FOLDER, "stage2_llm.jsonl")
NUM_GAMES = 200
MAX_PIECES = 20

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Question Templates ---

# VISUAL: Piece identification
VISUAL_PIECE_QUESTIONS = [
    "What piece is on {square}?",
    "Identify the piece at {square}.",
    "Which piece occupies {square}?",
    "What is located on square {square}?"
]

# VISUAL: Color identification
VISUAL_COLOR_QUESTIONS = [
    "What color is the piece on {square}?",
    "Is the piece on {square} white or black?",
    "Which player has a piece on {square}?",
    "What is the color of the piece at {square}?"
]

# VISUAL: Occupancy check
VISUAL_EXISTS_QUESTIONS = [
    "Is there a piece on {square}?",
    "Is {square} occupied?",
    "Does {square} have a piece?",
    "Is square {square} empty or occupied?"
]

# Board state questions
BOARD_STATE_QUESTIONS = [
    "Describe the complete board state row by row from rank 8 to rank 1.",
    "What is the current board position, rank by rank from 8 to 1?",
    "List all pieces on the board by rank, starting from rank 8 down to rank 1.",
    "Describe the board state from rank 8 to rank 1."
]

# Row-specific questions
ROW_QUESTIONS = [
    "What pieces are on rank {rank}?",
    "Describe the pieces on rank {rank}.",
    "List what's on rank {rank}.",
    "What is the current state of rank {rank}?"
]

# NEW: Column-specific questions
COLUMN_QUESTIONS = [
    "What pieces are on file {file}?",
    "Describe the pieces on file {file}.",
    "List what's on file {file}.",
    "What is the current state of file {file}?"
]

# NEW: Diagonal questions
DIAGONAL_QUESTIONS = [
    "What pieces are on the {diagonal} diagonal?",
    "Describe the pieces on the {diagonal} diagonal.",
    "List what's on the {diagonal} diagonal.",
    "What is the current state of the {diagonal} diagonal?"
]

# Spatial attack questions
ATTACK_QUESTIONS = [
    "Which {color} pieces can attack {square}?",
    "What {color} pieces are attacking {square}?",
    "Which {color} pieces have {square} in their attack range?",
    "What {color} pieces can reach {square} in one move?"
]

# Counting questions for specific piece types (max 8)
COUNTING_PAWNS_WHITE = [
    "How many white pawns are on the board?",
    "Count the white pawns.",
    "How many pawns does white have?",
    "What is the number of white pawns?"
]

COUNTING_PAWNS_BLACK = [
    "How many black pawns are on the board?",
    "Count the black pawns.",
    "How many pawns does black have?",
    "What is the number of black pawns?"
]

COUNTING_KNIGHTS_WHITE = [
    "How many white knights are on the board?",
    "Count the white knights.",
    "How many knights does white have?",
    "What is the number of white knights?"
]

COUNTING_KNIGHTS_BLACK = [
    "How many black knights are on the board?",
    "Count the black knights.",
    "How many knights does black have?",
    "What is the number of black knights?"
]

COUNTING_BISHOPS_WHITE = [
    "How many white bishops are on the board?",
    "Count the white bishops.",
    "How many bishops does white have?",
    "What is the number of white bishops?"
]

COUNTING_BISHOPS_BLACK = [
    "How many black bishops are on the board?",
    "Count the black bishops.",
    "How many bishops does black have?",
    "What is the number of black bishops?"
]

COUNTING_ROOKS_WHITE = [
    "How many white rooks are on the board?",
    "Count the white rooks.",
    "How many rooks does white have?",
    "What is the number of white rooks?"
]

COUNTING_ROOKS_BLACK = [
    "How many black rooks are on the board?",
    "Count the black rooks.",
    "How many rooks does black have?",
    "What is the number of black rooks?"
]

# Check questions
CHECK_QUESTIONS = [
    "Is either king in check?",
    "Is any king under attack?",
    "Is there a check on the board?",
    "Are either of the kings in check?"
]

# STRATEGY: Best move (with context)
BEST_MOVE_QUESTIONS = [
    "This is a chess position after {prev_color} played {prev_move}. It's {player}'s turn to move. What is the best move?",
    "In this position, {prev_color} just played {prev_move}. {player_cap} to move. What is the best move?",
    "After {prev_move} by {prev_color}, it's {player}'s turn. What should {player} play?",
    "This position arose after {prev_color}'s {prev_move}. What is the best move for {player}?"
]

# STRATEGY: Material balance
MATERIAL_QUESTIONS = [
    "Which side has more material?",
    "Who is ahead in material?",
    "Who has more material?",
    "What is the material balance?"
]

# --- Helper Functions ---

def sanitize_move(move):
    """Sanitize move for filename matching."""
    return move.replace("+", "").replace("#", "").replace("/", "").replace("=", "")

def count_total_pieces(board):
    """Count total pieces on board."""
    return sum(1 for sq in chess.SQUARES if board.piece_at(sq) is not None)

def count_pieces(board, piece_type=None, color=None):
    """Count pieces on board by type and/or color."""
    count = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            if piece_type and piece.piece_type != piece_type:
                continue
            if color is not None and piece.color != color:
                continue
            count += 1
    return count

def get_piece_name(piece):
    """Get standardized piece name."""
    piece_names = {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
        chess.KING: "king"
    }
    color = "white" if piece.color == chess.WHITE else "black"
    return f"{color}-{piece_names[piece.piece_type]}"

def get_rank_state(board, rank):
    """Get textual representation of a rank (row)."""
    pieces = []
    for file in range(8):  # a-h (0-7)
        square = chess.square(file, rank - 1)  # rank 1 = index 0
        piece = board.piece_at(square)
        if piece:
            pieces.append(get_piece_name(piece))
        else:
            pieces.append("empty")
    return " ".join(pieces)

def get_file_state(board, file_letter):
    """Get textual representation of a file (column)."""
    file_index = ord(file_letter) - ord('a')  # a=0, b=1, ..., h=7
    pieces = []
    for rank in range(8, 0, -1):  # 8 down to 1
        square = chess.square(file_index, rank - 1)
        piece = board.piece_at(square)
        if piece:
            pieces.append(get_piece_name(piece))
        else:
            pieces.append("empty")
    return " ".join(pieces)

def get_diagonal_state(board, diagonal_name):
    """Get textual representation of a diagonal."""
    # Define the two main diagonals
    if diagonal_name == "a1-h8":
        squares = [chess.A1, chess.B2, chess.C3, chess.D4, chess.E5, chess.F6, chess.G7, chess.H8]
    elif diagonal_name == "a8-h1":
        squares = [chess.A8, chess.B7, chess.C6, chess.D5, chess.E4, chess.F3, chess.G2, chess.H1]
    else:
        return ""
    
    pieces = []
    for square in squares:
        piece = board.piece_at(square)
        if piece:
            pieces.append(get_piece_name(piece))
        else:
            pieces.append("empty")
    return " ".join(pieces)

def get_full_board_state(board):
    """Get textual representation of entire board, rank by rank."""
    lines = []
    for rank in range(8, 0, -1):  # 8 down to 1
        rank_state = get_rank_state(board, rank)
        lines.append(f"Rank {rank}: {rank_state}")
    return "\n".join(lines)

def get_attacking_pieces(board, target_square, color):
    """Get list of pieces of given color that can attack target square."""
    attackers = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color:
            if board.is_attacked_by(color, target_square):
                if target_square in board.attacks(square):
                    attackers.append(f"{get_piece_name(piece)} on {chess.square_name(square)}")
    return attackers

# --- Answer Variation Functions ---

def get_piece_answer_varied(piece):
    """Get varied answer for piece identification."""
    piece_names = {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
        chess.KING: "king"
    }
    color = "white" if piece.color == chess.WHITE else "black"
    piece_name = piece_names[piece.piece_type]
    
    variations = [
        f"{color} {piece_name}",
        f"a {color} {piece_name}",
        f"{piece_name} ({color})",
        f"{color}'s {piece_name}"
    ]
    return random.choice(variations)

def get_color_answer_varied(piece):
    """Get varied answer for color identification."""
    color = "white" if piece.color == chess.WHITE else "black"
    variations = [
        color,
        f"the piece is {color}",
        color.capitalize()
    ]
    return random.choice(variations)

def get_exists_answer_varied(exists):
    """Get varied answer for existence check."""
    if exists:
        variations = ["yes", "Yes", "occupied", "yes, occupied"]
        return random.choice(variations)
    else:
        variations = ["no", "No", "empty", "no, empty"]
        return random.choice(variations)

def get_check_answer_varied(is_check):
    """Get varied answer for check status."""
    if is_check:
        variations = ["yes", "Yes", "yes, one king is in check", "check"]
        return random.choice(variations)
    else:
        variations = ["no", "No", "both kings are safe", "no checks"]
        return random.choice(variations)

def calculate_material(board):
    """Calculate material balance."""
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    white_material = sum(piece_values[p.piece_type] for sq in chess.SQUARES 
                         if (p := board.piece_at(sq)) and p.color == chess.WHITE)
    black_material = sum(piece_values[p.piece_type] for sq in chess.SQUARES 
                         if (p := board.piece_at(sq)) and p.color == chess.BLACK)
    
    if white_material > black_material:
        return "white"
    elif black_material > white_material:
        return "black"
    else:
        return "equal"

def get_material_answer_varied(material):
    """Get varied answer for material balance."""
    if material == "equal":
        variations = ["equal", "Equal", "balanced", "the material is equal"]
    elif material == "white":
        variations = ["white", "White", "white is ahead", "White has more material"]
    else:
        variations = ["black", "Black", "black is ahead", "Black has more material"]
    return random.choice(variations)

# --- Question Generation Functions ---

def generate_visual_questions_board(board, img_path, ply):
    """Generate 9 VISUAL questions for Stage 1 - VERSION WITH BOARD STATE.
    
    Structure:
    - 1 full board state question
    - 2 row-specific questions
    - 4 basic visual questions (piece ID, color, exists)
    - 2 counting questions (specific piece types, under 8)
    """
    questions = []
    
    occupied = [sq for sq in chess.SQUARES if board.piece_at(sq) is not None]
    all_squares = list(chess.SQUARES)
    
    if len(occupied) < 3:
        return []
    
    random.shuffle(occupied)
    random.shuffle(all_squares)
    
    # Question 1 - Full board state
    q1_text = random.choice(BOARD_STATE_QUESTIONS)
    board_state = get_full_board_state(board)
    
    questions.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img_path.replace("\\", "/")},
                {"type": "text", "text": q1_text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": board_state}
            ]}
        ]
    })
    
    # Questions 2-3 - Two row-specific questions
    random_ranks = random.sample(range(1, 9), 2)
    
    for rank in random_ranks:
        q_text = random.choice(ROW_QUESTIONS).format(rank=rank)
        rank_state = get_rank_state(board, rank)
        
        questions.append({
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": img_path.replace("\\", "/")},
                    {"type": "text", "text": q_text}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": rank_state}
                ]}
            ]
        })
    
    # Questions 4-7 - Basic visual questions (4 total)
    
    # Q4: Piece identification
    sq1 = occupied[0]
    piece1 = board.piece_at(sq1)
    q4_text = random.choice(VISUAL_PIECE_QUESTIONS).format(square=chess.square_name(sq1))
    questions.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img_path.replace("\\", "/")},
                {"type": "text", "text": q4_text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": get_piece_answer_varied(piece1)}
            ]}
        ]
    })
    
    # Q5: Color identification
    sq2 = occupied[1]
    piece2 = board.piece_at(sq2)
    q5_text = random.choice(VISUAL_COLOR_QUESTIONS).format(square=chess.square_name(sq2))
    questions.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img_path.replace("\\", "/")},
                {"type": "text", "text": q5_text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": get_color_answer_varied(piece2)}
            ]}
        ]
    })
    
    # Q6: Occupancy check
    sq3 = all_squares[0]
    piece3 = board.piece_at(sq3)
    q6_text = random.choice(VISUAL_EXISTS_QUESTIONS).format(square=chess.square_name(sq3))
    questions.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img_path.replace("\\", "/")},
                {"type": "text", "text": q6_text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": get_exists_answer_varied(piece3 is not None)}
            ]}
        ]
    })
    
    # Q7: Another piece identification
    sq4 = occupied[2]
    piece4 = board.piece_at(sq4)
    q7_text = random.choice(VISUAL_PIECE_QUESTIONS).format(square=chess.square_name(sq4))
    questions.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img_path.replace("\\", "/")},
                {"type": "text", "text": q7_text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": get_piece_answer_varied(piece4)}
            ]}
        ]
    })
    
    # Questions 8-9 - Two counting questions (specific pieces, max 8)
    counting_options = [
        (COUNTING_PAWNS_WHITE, chess.PAWN, chess.WHITE, "pawns"),
        (COUNTING_PAWNS_BLACK, chess.PAWN, chess.BLACK, "pawns"),
        (COUNTING_KNIGHTS_WHITE, chess.KNIGHT, chess.WHITE, "knights"),
        (COUNTING_KNIGHTS_BLACK, chess.KNIGHT, chess.BLACK, "knights"),
        (COUNTING_BISHOPS_WHITE, chess.BISHOP, chess.WHITE, "bishops"),
        (COUNTING_BISHOPS_BLACK, chess.BISHOP, chess.BLACK, "bishops"),
        (COUNTING_ROOKS_WHITE, chess.ROOK, chess.WHITE, "rooks"),
        (COUNTING_ROOKS_BLACK, chess.ROOK, chess.BLACK, "rooks"),
    ]
    
    selected_counting = random.sample(counting_options, 2)
    
    for q_templates, piece_type, color, piece_name in selected_counting:
        count = count_pieces(board, piece_type, color)
        q_text = random.choice(q_templates)
        
        if random.random() < 0.7:
            answer = str(count)
        else:
            answer = f"{count} {piece_name}" if count != 1 else f"1 {piece_name[:-1]}"
        
        questions.append({
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": img_path.replace("\\", "/")},
                    {"type": "text", "text": q_text}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": answer}
                ]}
            ]
        })
    
    return questions

def generate_visual_questions_rcd(board, img_path, ply):
    """Generate 10 VISUAL questions for Stage 1 - VERSION WITH ROW/COLUMN/DIAGONAL.
    
    Structure:
    - 2 row-specific questions (UPDATED from 1 to 2)
    - 1 column-specific question
    - 1 diagonal-specific question
    - 4 basic visual questions (piece ID, color, exists)
    - 2 counting questions (specific piece types, under 8)
    """
    questions = []
    
    occupied = [sq for sq in chess.SQUARES if board.piece_at(sq) is not None]
    all_squares = list(chess.SQUARES)
    
    if len(occupied) < 3:
        return []
    
    random.shuffle(occupied)
    random.shuffle(all_squares)
    
    # Questions 1-2 - TWO Row-specific questions (CHANGED from 1 to 2)
    random_ranks = random.sample(range(1, 9), 2)  # Select 2 different ranks
    
    for rank in random_ranks:
        q_text = random.choice(ROW_QUESTIONS).format(rank=rank)
        rank_state = get_rank_state(board, rank)
        
        questions.append({
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": img_path.replace("\\", "/")},
                    {"type": "text", "text": q_text}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": rank_state}
                ]}
            ]
        })
    
    # Question 3 - Column-specific
    random_file = random.choice(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    q3_text = random.choice(COLUMN_QUESTIONS).format(file=random_file)
    file_state = get_file_state(board, random_file)
    
    questions.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img_path.replace("\\", "/")},
                {"type": "text", "text": q3_text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": file_state}
            ]}
        ]
    })
    
    # Question 4 - Diagonal-specific
    random_diagonal = random.choice(["a1-h8", "a8-h1"])
    q4_text = random.choice(DIAGONAL_QUESTIONS).format(diagonal=random_diagonal)
    diagonal_state = get_diagonal_state(board, random_diagonal)
    
    questions.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img_path.replace("\\", "/")},
                {"type": "text", "text": q4_text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": diagonal_state}
            ]}
        ]
    })
    
    # Questions 5-8 - Basic visual questions (4 total)
    
    # Q5: Piece identification
    sq1 = occupied[0]
    piece1 = board.piece_at(sq1)
    q5_text = random.choice(VISUAL_PIECE_QUESTIONS).format(square=chess.square_name(sq1))
    questions.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img_path.replace("\\", "/")},
                {"type": "text", "text": q5_text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": get_piece_answer_varied(piece1)}
            ]}
        ]
    })
    
    # Q6: Color identification
    sq2 = occupied[1]
    piece2 = board.piece_at(sq2)
    q6_text = random.choice(VISUAL_COLOR_QUESTIONS).format(square=chess.square_name(sq2))
    questions.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img_path.replace("\\", "/")},
                {"type": "text", "text": q6_text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": get_color_answer_varied(piece2)}
            ]}
        ]
    })
    
    # Q7: Occupancy check
    sq3 = all_squares[0]
    piece3 = board.piece_at(sq3)
    q7_text = random.choice(VISUAL_EXISTS_QUESTIONS).format(square=chess.square_name(sq3))
    questions.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img_path.replace("\\", "/")},
                {"type": "text", "text": q7_text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": get_exists_answer_varied(piece3 is not None)}
            ]}
        ]
    })
    
    # Q8: Another piece identification
    sq4 = occupied[2]
    piece4 = board.piece_at(sq4)
    q8_text = random.choice(VISUAL_PIECE_QUESTIONS).format(square=chess.square_name(sq4))
    questions.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img_path.replace("\\", "/")},
                {"type": "text", "text": q8_text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": get_piece_answer_varied(piece4)}
            ]}
        ]
    })
    
    # Questions 9-10 - Two counting questions (specific pieces, max 8)
    counting_options = [
        (COUNTING_PAWNS_WHITE, chess.PAWN, chess.WHITE, "pawns"),
        (COUNTING_PAWNS_BLACK, chess.PAWN, chess.BLACK, "pawns"),
        (COUNTING_KNIGHTS_WHITE, chess.KNIGHT, chess.WHITE, "knights"),
        (COUNTING_KNIGHTS_BLACK, chess.KNIGHT, chess.BLACK, "knights"),
        (COUNTING_BISHOPS_WHITE, chess.BISHOP, chess.WHITE, "bishops"),
        (COUNTING_BISHOPS_BLACK, chess.BISHOP, chess.BLACK, "bishops"),
        (COUNTING_ROOKS_WHITE, chess.ROOK, chess.WHITE, "rooks"),
        (COUNTING_ROOKS_BLACK, chess.ROOK, chess.BLACK, "rooks"),
    ]
    
    selected_counting = random.sample(counting_options, 2)
    
    for q_templates, piece_type, color, piece_name in selected_counting:
        count = count_pieces(board, piece_type, color)
        q_text = random.choice(q_templates)
        
        if random.random() < 0.7:
            answer = str(count)
        else:
            answer = f"{count} {piece_name}" if count != 1 else f"1 {piece_name[:-1]}"
        
        questions.append({
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": img_path.replace("\\", "/")},
                    {"type": "text", "text": q_text}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": answer}
                ]}
            ]
        })
    
    return questions

def generate_strategy_questions(board, next_san, prev_san, img_path):
    """Generate 7 STRATEGY questions for Stage 2 (regular positions).
    
    Structure:
    - 3 visual questions (from Stage 1 types)
    - 4 strategy questions (best move with context, check, material, attacks)
    """
    questions = []
    
    player = "white" if board.turn == chess.WHITE else "black"
    player_cap = player.capitalize()
    prev_player = "black" if board.turn == chess.WHITE else "white"
    
    # Questions 1-3 - Visual grounding questions
    
    occupied = [sq for sq in chess.SQUARES if board.piece_at(sq) is not None]
    all_squares = list(chess.SQUARES)
    
    if len(occupied) < 2:
        return []
    
    random.shuffle(occupied)
    random.shuffle(all_squares)
    
    # Visual Q1: Piece identification
    sq1 = occupied[0]
    piece1 = board.piece_at(sq1)
    vq1_text = random.choice(VISUAL_PIECE_QUESTIONS).format(square=chess.square_name(sq1))
    questions.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img_path.replace("\\", "/")},
                {"type": "text", "text": vq1_text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": get_piece_answer_varied(piece1)}
            ]}
        ]
    })
    
    # Visual Q2: Occupancy check
    sq2 = all_squares[0]
    piece2 = board.piece_at(sq2)
    vq2_text = random.choice(VISUAL_EXISTS_QUESTIONS).format(square=chess.square_name(sq2))
    questions.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img_path.replace("\\", "/")},
                {"type": "text", "text": vq2_text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": get_exists_answer_varied(piece2 is not None)}
            ]}
        ]
    })
    
    # Visual Q3: Counting (specific piece type, under 8)
    counting_options = [
        (COUNTING_KNIGHTS_WHITE, chess.KNIGHT, chess.WHITE, "knights"),
        (COUNTING_KNIGHTS_BLACK, chess.KNIGHT, chess.BLACK, "knights"),
        (COUNTING_BISHOPS_WHITE, chess.BISHOP, chess.WHITE, "bishops"),
        (COUNTING_BISHOPS_BLACK, chess.BISHOP, chess.BLACK, "bishops"),
        (COUNTING_ROOKS_WHITE, chess.ROOK, chess.WHITE, "rooks"),
        (COUNTING_ROOKS_BLACK, chess.ROOK, chess.BLACK, "rooks"),
        (COUNTING_PAWNS_WHITE, chess.PAWN, chess.WHITE, "pawns"),
        (COUNTING_PAWNS_BLACK, chess.PAWN, chess.BLACK, "pawns"),
    ]
    
    q_templates, piece_type, color, piece_name = random.choice(counting_options)
    count = count_pieces(board, piece_type, color)
    vq3_text = random.choice(q_templates)
    
    if random.random() < 0.7:
        answer = str(count)
    else:
        answer = f"{count} {piece_name}" if count != 1 else f"1 {piece_name[:-1]}"
    
    questions.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img_path.replace("\\", "/")},
                {"type": "text", "text": vq3_text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": answer}
            ]}
        ]
    })
    
    # Strategy Questions 4-7
    
    # Strategy Q4: Best move (with context from previous move)
    q4_template = random.choice(BEST_MOVE_QUESTIONS)
    q4_text = q4_template.format(
        prev_color=prev_player,
        prev_move=prev_san,
        player=player,
        player_cap=player_cap
    )
    
    if random.random() < 0.8:
        answer = next_san
    else:
        answer = f"The best move is {next_san}"
    
    questions.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img_path.replace("\\", "/")},
                {"type": "text", "text": q4_text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": answer}
            ]}
        ]
    })
    
    # Strategy Q5: Check status
    is_check = board.is_check()
    q5_text = random.choice(CHECK_QUESTIONS)
    questions.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img_path.replace("\\", "/")},
                {"type": "text", "text": q5_text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": get_check_answer_varied(is_check)}
            ]}
        ]
    })
    
    # Strategy Q6: Material balance
    material = calculate_material(board)
    q6_text = random.choice(MATERIAL_QUESTIONS)
    questions.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img_path.replace("\\", "/")},
                {"type": "text", "text": q6_text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": get_material_answer_varied(material)}
            ]}
        ]
    })
    
    # Strategy Q7: Spatial attack question
    attack_color = chess.WHITE if random.random() < 0.5 else chess.BLACK
    color_name = "white" if attack_color == chess.WHITE else "black"
    
    target_square = random.choice(all_squares)
    attackers = get_attacking_pieces(board, target_square, attack_color)
    
    q7_text = random.choice(ATTACK_QUESTIONS).format(
        color=color_name,
        square=chess.square_name(target_square)
    )
    
    if attackers:
        if len(attackers) == 1:
            answer = attackers[0]
        else:
            answer = ", ".join(attackers)
    else:
        answer_variations = ["none", "no pieces", f"no {color_name} pieces can attack {chess.square_name(target_square)}"]
        answer = random.choice(answer_variations)
    
    questions.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img_path.replace("\\", "/")},
                {"type": "text", "text": q7_text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": answer}
            ]}
        ]
    })
    
    return questions

def generate_endgame_questions(board, game_result, img_path):
    """Generate endgame questions for final position (Stage 2)."""
    questions = []
    
    # Question 1: Check status
    is_check = board.is_check()
    q1_text = random.choice(CHECK_QUESTIONS)
    questions.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img_path.replace("\\", "/")},
                {"type": "text", "text": q1_text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": get_check_answer_varied(is_check)}
            ]}
        ]
    })
    
    # Question 2: Who won
    who_won_questions = [
        "Who won this game?",
        "What was the result of this game?",
        "Who is the winner of this game?",
        "What is the outcome of this game?"
    ]
    
    q2_text = random.choice(who_won_questions)
    
    if game_result == "1-0":
        winner_variations = ["white", "White", "white won", "White wins"]
        winner = random.choice(winner_variations)
    elif game_result == "0-1":
        winner_variations = ["black", "Black", "black won", "Black wins"]
        winner = random.choice(winner_variations)
    else:
        winner_variations = ["draw", "Draw", "it's a draw", "the game was drawn"]
        winner = random.choice(winner_variations)
    
    questions.append({
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": img_path.replace("\\", "/")},
                {"type": "text", "text": q2_text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": winner}
            ]}
        ]
    })
    
    return questions

# --- Main Processing ---
visual_board_rows = []
visual_rcd_rows = []
strategy_rows = []
games_processed = 0
positions_processed = 0
skipped_no_image = 0
skipped_no_pieces = 0
skipped_too_many_pieces = 0

games_used = set()

# Get first 200 PGN files
print("Loading PGN files...")
all_pgn_files = [f for f in os.listdir(PGN_FOLDER) if f.endswith(".pgn")]

pgn_files = sorted(
    all_pgn_files,
    key=lambda x: int(x.replace("game", "").replace(".pgn", ""))
)[:NUM_GAMES]

# Verify game selection
print("=" * 70)
print("CHESS VLM DATASET GENERATION - 3 OUTPUT FILES VERSION")
print("=" * 70)
print(f"Stage 1A: Visual (Board State) - stage1_visual_board.jsonl")
print(f"Stage 1B: Visual (Row/Column/Diagonal) - stage1_visual_rcd.jsonl")
print(f"Stage 2: Strategy + Visual Grounding - stage2_llm.jsonl")
print("=" * 70)
print(f"Configuration:")
print(f"  - Max pieces per position: {MAX_PIECES}")
print(f"  - PGN files found: {len(all_pgn_files)}")
print(f"  - Selected for processing: {len(pgn_files)}")
print(f"  - Game range: {pgn_files[0]} to {pgn_files[-1]}")
print("=" * 70)

print(f"\nProcessing {len(pgn_files)} games...")
print("=" * 70)

for pgn_file in pgn_files:
    game_id = pgn_file.replace(".pgn", "")
    game_num = int(game_id.replace("game", ""))
    games_used.add(game_num)
    
    pgn_path = os.path.join(PGN_FOLDER, pgn_file)
    
    with open(pgn_path) as f:
        game = chess.pgn.read_game(f)
        if game is None:
            continue
        
        game_result = game.headers.get("Result", "1/2-1/2")
        
        board = game.board()
        moves = list(game.mainline_moves())
        ply = 1
        prev_san = None
        
        for i in range(len(moves)):
            move = moves[i]
            san_move = board.san(move)
            
            img_name = f"{game_id}_{sanitize_move(san_move)}_ply{ply}.jpg"
            img_path = os.path.join(IMG_FOLDER, img_name)
            
            if not os.path.exists(img_path):
                prev_san = san_move
                board.push(move)
                ply += 1
                skipped_no_image += 1
                continue
            
            board.push(move)
            
            total_pieces = count_total_pieces(board)
            if total_pieces > MAX_PIECES:
                prev_san = san_move
                ply += 1
                skipped_too_many_pieces += 1
                continue
            
            # Generate VISUAL questions - BOTH VERSIONS
            visual_board_qs = generate_visual_questions_board(board, img_path, ply)
            visual_rcd_qs = generate_visual_questions_rcd(board, img_path, ply)
            
            if len(visual_board_qs) < 9 or len(visual_rcd_qs) < 10:
                prev_san = san_move
                ply += 1
                skipped_no_pieces += 1
                continue
            
            visual_board_rows.extend(visual_board_qs)
            visual_rcd_rows.extend(visual_rcd_qs)
            
            # Generate STRATEGY questions (Stage 2)
            is_last_move = (i == len(moves) - 1)
            
            if is_last_move:
                strategy_qs = generate_endgame_questions(board, game_result, img_path)
            else:
                next_move = moves[i + 1]
                next_san = board.san(next_move)
                
                if prev_san:
                    strategy_qs = generate_strategy_questions(board, next_san, prev_san, img_path)
                else:
                    strategy_qs = generate_strategy_questions(board, next_san, "the opening", img_path)
            
            strategy_rows.extend(strategy_qs)
            
            positions_processed += 1
            prev_san = san_move
            ply += 1
        
        games_processed += 1
        if games_processed % 10 == 0:
            print(f"Processed {games_processed}/{len(pgn_files)} games | "
                  f"Positions: {positions_processed} | "
                  f"Board: {len(visual_board_rows)} | RCD: {len(visual_rcd_rows)} | Strategy: {len(strategy_rows)}")

# --- Save All 3 JSONL Files ---
print("\n" + "=" * 70)
print("Saving datasets...")

# Save Stage 1A: Visual Board
with open(JSONL_VISUAL_BOARD, "w", encoding="utf-8") as f:
    for row in visual_board_rows:
        json.dump(row, f)
        f.write("\n")
print(f"✅ Stage 1A (Visual - Board State) saved to: {JSONL_VISUAL_BOARD}")

# Save Stage 1B: Visual RCD
with open(JSONL_VISUAL_RCD, "w", encoding="utf-8") as f:
    for row in visual_rcd_rows:
        json.dump(row, f)
        f.write("\n")
print(f"✅ Stage 1B (Visual - Row/Column/Diagonal) saved to: {JSONL_VISUAL_RCD}")

# Save Stage 2: Strategy
with open(JSONL_STRATEGY, "w", encoding="utf-8") as f:
    for row in strategy_rows:
        json.dump(row, f)
        f.write("\n")
print(f"✅ Stage 2 (Strategy) saved to: {JSONL_STRATEGY}")

# --- Statistics ---
print("\n" + "=" * 70)
print("DATASET GENERATION COMPLETE - 3 FILES!")
print("=" * 70)
print(f"Games processed: {games_processed}")
print(f"Game numbers used: {min(games_used)} to {max(games_used)}")
print(f"Positions processed: {positions_processed}")

print(f"\n📁 STAGE 1A: Visual (Board State) - stage1_visual_board.jsonl")
print(f"  Total samples: {len(visual_board_rows)}")
print(f"  Expected per position: 9")
print(f"  Actual per position: {len(visual_board_rows) / positions_processed:.1f}")
print(f"  Structure:")
print(f"    - 1 Full board state question")
print(f"    - 2 Row-specific questions")
print(f"    - 4 Basic visual (piece ID, color, exists)")
print(f"    - 2 Counting questions")

print(f"\n📁 STAGE 1B: Visual (RCD) - stage1_visual_rcd.jsonl")
print(f"  Total samples: {len(visual_rcd_rows)}")
print(f"  Expected per position: 10")
print(f"  Actual per position: {len(visual_rcd_rows) / positions_processed:.1f}")
print(f"  Structure:")
print(f"    - 2 Row-specific questions (UPDATED)")
print(f"    - 1 Column-specific question")
print(f"    - 1 Diagonal-specific question")
print(f"    - 4 Basic visual (piece ID, color, exists)")
print(f"    - 2 Counting questions")

print(f"\n📁 STAGE 2: Strategy - stage2_llm.jsonl")
print(f"  Total samples: {len(strategy_rows)}")
print(f"  Expected per regular position: 7")
print(f"  Expected per endgame position: 2")
regular_positions = positions_processed - games_processed
print(f"  Estimated breakdown:")
print(f"    - Visual grounding: ~{regular_positions * 3}")
print(f"    - Best move (with context): ~{regular_positions}")
print(f"    - Check status: ~{positions_processed}")
print(f"    - Material balance: ~{regular_positions}")
print(f"    - Spatial attacks: ~{regular_positions}")
print(f"    - Game outcome: ~{games_processed}")

print(f"\nFiltering Statistics:")
print(f"  - Positions with ≤{MAX_PIECES} pieces: {positions_processed}")
print(f"  - Skipped (too many pieces): {skipped_too_many_pieces}")
print(f"  - Skipped (no image): {skipped_no_image}")
print(f"  - Skipped (not enough pieces): {skipped_no_pieces}")
print("=" * 70)

# --- Sample Display ---
print("\n" + "=" * 70)
print("SAMPLE ENTRIES FROM EACH FILE")
print("=" * 70)

print("\n--- STAGE 1A (VISUAL - BOARD STATE) SAMPLES ---")
with open(JSONL_VISUAL_BOARD, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"\nShowing 3 random samples from {len(lines)} total:")
print("-" * 70)
for i, sample_line in enumerate(random.sample(lines, min(3, len(lines))), 1):
    obj = json.loads(sample_line)
    user_msg = obj["messages"][0]["content"]
    assistant_msg = obj["messages"][1]["content"]
    
    question = next((item["text"] for item in user_msg if item["type"] == "text"), "")
    answer = next((item["text"] for item in assistant_msg if item["type"] == "text"), "")
    image = next((item["image"] for item in user_msg if item["type"] == "image"), "")
    
    if len(answer) > 150:
        answer = answer[:150] + "..."
    
    print(f"\nBoard Sample {i}:")
    print(f"Image: {os.path.basename(image)}")
    print(f"Q: {question}")
    print(f"A: {answer}")

print("\n--- STAGE 1B (VISUAL - ROW/COLUMN/DIAGONAL) SAMPLES ---")
with open(JSONL_VISUAL_RCD, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"\nShowing 3 random samples from {len(lines)} total:")
print("-" * 70)
for i, sample_line in enumerate(random.sample(lines, min(3, len(lines))), 1):
    obj = json.loads(sample_line)
    user_msg = obj["messages"][0]["content"]
    assistant_msg = obj["messages"][1]["content"]
    
    question = next((item["text"] for item in user_msg if item["type"] == "text"), "")
    answer = next((item["text"] for item in assistant_msg if item["type"] == "text"), "")
    image = next((item["image"] for item in user_msg if item["type"] == "image"), "")
    
    print(f"\nRCD Sample {i}:")
    print(f"Image: {os.path.basename(image)}")
    print(f"Q: {question}")
    print(f"A: {answer}")

print("\n--- STAGE 2 (STRATEGY) SAMPLES ---")
with open(JSONL_STRATEGY, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"\nShowing 3 random samples from {len(lines)} total:")
print("-" * 70)
for i, sample_line in enumerate(random.sample(lines, min(3, len(lines))), 1):
    obj = json.loads(sample_line)
    user_msg = obj["messages"][0]["content"]
    assistant_msg = obj["messages"][1]["content"]
    
    question = next((item["text"] for item in user_msg if item["type"] == "text"), "")
    answer = next((item["text"] for item in assistant_msg if item["type"] == "text"), "")
    image = next((item["image"] for item in user_msg if item["type"] == "image"), "")
    
    print(f"\nStrategy Sample {i}:")
    print(f"Image: {os.path.basename(image)}")
    print(f"Q: {question}")
    print(f"A: {answer}")

print("\n" + "=" * 70)
print("✅ ALL DONE! 3 JSONL files ready for testing!")
print("=" * 70)
print("\nYOU NOW HAVE 3 OUTPUT FILES:")
print("  1️⃣  stage1_visual_board.jsonl (9 Q/position - with full board state)")
print("  2️⃣  stage1_visual_rcd.jsonl (10 Q/position - 2 rows + 1 column + 1 diagonal)")
print("  3️⃣  stage2_llm.jsonl (strategy + visual grounding)")
print("\nNEXT STEPS:")
print("  - Test BOTH Stage 1 versions to see which works better")
print("  - Use the same Stage 2 file for both experiments")
print("  - Compare: does board state or RCD produce better results?")
print("=" * 70)