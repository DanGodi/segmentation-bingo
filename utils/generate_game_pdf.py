import pandas as pd
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from PIL import Image
import re
import textwrap

# --- Helper Functions ---

def load_data(stats_path, cards_path):
    """Loads stats and cards data."""
    stats_path = Path(stats_path)
    cards_path = Path(cards_path)
    
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    if not cards_path.exists():
        raise FileNotFoundError(f"Cards file not found: {cards_path}")
        
    df = pd.read_csv(stats_path)
    df["n_objects"] = df["n_objects"].fillna(0).astype(int)
    
    # Pivot to get Image x Feature matrix
    counts_matrix = df.pivot_table(index="image", columns="feature", values="n_objects", fill_value=0)
    
    with open(cards_path, "r") as f:
        cards_data = json.load(f)
        
    return counts_matrix, cards_data

def parse_event(event_str, row_data):
    """
    Parses an event string (e.g., "More than 2 Pools") and checks if it's true for a given image row.
    row_data is a Series: index=feature_name, value=count
    """
    # 1. Contains {feat}
    if event_str.startswith("Contains "):
        feat = event_str[9:]
        if feat in row_data:
            return row_data[feat] > 0
            
    # 2. More than {n} {feat}s
    # Regex: More than (\d+) (.+)s
    if event_str.startswith("More than "):
        match = re.match(r"More than (\d+) (.+)", event_str)
        if match:
            n = int(match.group(1))
            rest = match.group(2)
            # The generator added 's' to the end. Let's try removing it.
            for col in row_data.index:
                # Construct what the string WOULD be
                if rest == f"{col}s":
                    return row_data[col] > n
    
    # 3. Exactly {n} {feat}
    if event_str.startswith("Exactly "):
        match = re.match(r"Exactly (\d+) (.+)", event_str)
        if match:
            n = int(match.group(1))
            rest = match.group(2)
            # Handle plural 's' if n > 1
            suffix = "s" if n > 1 else ""
            for col in row_data.index:
                if rest == f"{col}{suffix}":
                    return row_data[col] == n
                    
    # Fallback: If we can't parse, assume False (or print warning)
    # print(f"Warning: Could not parse event '{event_str}'")
    return False

def simulate_game(image_order, counts_matrix, cards_data):
    """
    Simulates the game with the given image order.
    Returns:
        winners: list of (card_id, turn_index)
        card_progress: dict {card_id: [squares_left_at_turn_0, squares_left_at_turn_1, ...]}
    """
    n_cards = len(cards_data)
    # Support variable card sizes: each card may have a different number of squares
    card_events = {c['card_id']: c['events'] for c in cards_data}
    card_progress = {c['card_id']: [len(c['events'])] for c in cards_data} # Starts with needed squares
    card_status = {c['card_id']: [False] * len(c['events']) for c in cards_data}
    
    winners = []
    
    for turn_idx, img_name in enumerate(image_order):
        # Get features for this image
        if img_name in counts_matrix.index:
            row = counts_matrix.loc[img_name]
            
            # Update all cards
            for c in cards_data:
                cid = c['card_id']
                events = c['events']
                
                # Check each event
                for i, event_str in enumerate(events):
                    if not card_status[cid][i]: # If not already marked
                        if parse_event(event_str, row):
                            card_status[cid][i] = True
                            
                # Record progress
                squares_left = len(events) - sum(card_status[cid])
                card_progress[cid].append(squares_left)
                
                # Check win
                if squares_left == 0:
                    # Check if already won?
                    # We want to know WHEN they won.
                    # If they just won this turn, add to winners
                    # But we might have multiple winners this turn.
                    # Check if they were already a winner?
                    # winners is list of (id, turn).
                    already_won = any(w[0] == cid for w in winners)
                    if not already_won:
                        winners.append((cid, turn_idx + 1))
                        
        else:
            # Image not in stats? Just record progress as same as last
            for cid in card_progress:
                last_val = card_progress[cid][-1]
                card_progress[cid].append(last_val)
                
    return winners, card_progress

def create_presentation(stats_path, cards_path, images_dir, output_pdf):
    print("Generating Bingo Presentation...")
    
    # 1. Load Data
    counts_matrix, cards_data = load_data(stats_path, cards_path)
    all_images = list(counts_matrix.index)
    
    # 2. Shuffle Images
    random.shuffle(all_images)
    print(f"Shuffled {len(all_images)} images.")
    
    # 3. Simulate Game to find stats
    winners, card_progress = simulate_game(all_images, counts_matrix, cards_data)
    
    # Identify podium
    first_place = winners[0] if len(winners) > 0 else None
    second_place = winners[1] if len(winners) > 1 else None
    third_place = winners[2] if len(winners) > 2 else None

    if first_place:
        print(f"The winner won on turn {first_place[1]}")
    else:
        print("No winners found in the simulated order.")
    if second_place:
        print(f"Second place won on turn {second_place[1]}")
    
    if third_place:
        print(f"Third place won on turn {third_place[1]}")
    
    # 4. Generate PDF
    with PdfPages(output_pdf) as pdf:
        
        # --- Title Slide ---
        plt.figure(figsize=(11.69, 8.27))
        plt.text(0.5, 0.6, "Satellite Bingo", ha='center', fontsize=40, weight='bold')
        plt.text(0.5, 0.4, "Get your cards ready!", ha='center', fontsize=20)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # --- Image Slides ---
        # We'll show images until the 3rd winner is found, plus a buffer?
        # Or just show all? Let's show all for now, or maybe limit to 50 if there are too many.
        
        for i, img_name in enumerate(all_images):
            img_path = images_dir / img_name
            
            plt.figure(figsize=(11.69, 8.27))
            
            # Title: Turn Number
            plt.suptitle(f"Turn #{i+1}", fontsize=24, weight='bold', y=0.95)
            
            # Image
            if img_path.exists():
                try:
                    img = Image.open(img_path)
                    plt.imshow(img)
                    plt.axis('off')
                except Exception as e:
                    plt.text(0.5, 0.5, f"Error loading image:\n{img_name}", ha='center')
                    plt.axis('off')
            else:
                # Try to find it in the index (maybe path mismatch)
                # The index in counts_matrix is the full path string from analyze_segment.py
                # But here we are iterating names?
                # Wait, counts_matrix index is "image" column from CSV.
                # In analyze_segment.py: "image": str(img_path) (absolute path)
                # So all_images contains absolute paths as strings.
                
                # Let's try to open it directly
                try:
                    img = Image.open(img_name)
                    plt.imshow(img)
                    plt.axis('off')
                except Exception:
                     plt.text(0.5, 0.5, f"Image not found:\n{img_name}", ha='center')
                     plt.axis('off')
                
            # Footer
            plt.figtext(0.5, 0.02, "Satellite Bingo", ha='center', fontsize=10, color='gray')
            
            pdf.savefig()
            plt.close()
            
            # Stop if we have 3 winners and passed some buffer?
            # Let's stop after 3rd winner + 5 turns
            if third_place and i > third_place[1] + 5:
                break
            
        # --- Winner Reveal Slide ---
        plt.figure(figsize=(11.69, 8.27))
        plt.text(0.5, 0.8, "GAME OVER!", ha='center', fontsize=40, weight='bold', color='red')
        plt.text(0.5, 0.6, "And the winners are...", ha='center', fontsize=30)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # --- Podium Slide ---
        plt.figure(figsize=(11.69, 8.27))
        
        places = [first_place, second_place, third_place]
        labels = ["1st Place", "2nd Place", "3rd Place"]
        colors = ['gold', 'silver', '#cd7f32'] # Bronze
        
        # Bar chart for podium
        # X coords
        x = [1, 0, 2] # Center, Left, Right
        heights = [3, 2, 1]
        
        for idx, place in enumerate(places):
            if place:
                card_id, turn = place
                # Draw bar
                plt.bar(x[idx], heights[idx], color=colors[idx], width=0.8)
                # Label
                plt.text(x[idx], heights[idx] + 0.1, f"Card #{card_id}\n(Turn {turn})", 
                         ha='center', fontsize=16, weight='bold')
                plt.text(x[idx], heights[idx]/2, labels[idx], ha='center', color='white', weight='bold')
                
        plt.xlim(-1, 3)
        plt.ylim(0, 4)
        plt.axis('off')
        plt.title("The Winners Podium", fontsize=30)
        pdf.savefig()
        plt.close()
        
        # --- Fun Stats: The Race ---
        plt.figure(figsize=(11.69, 8.27))

        # Plot progress of top 3 vs Average
        turns = range(len(all_images) + 1)

        # Calculate average progress
        all_progress = np.array([card_progress[c['card_id']] for c in cards_data], dtype=object)
        try:
            # Find max length
            max_len = max(len(p) for p in card_progress.values())
            # Pad with 0 (won)
            padded_progress = []
            for cid, prog in card_progress.items():
                if len(prog) < max_len:
                    prog = prog + [0] * (max_len - len(prog))
                padded_progress.append(prog)
            
            avg_needed = np.mean(padded_progress, axis=0)
        except Exception:
            avg_needed = []

        if len(avg_needed) > 0:
            plt.plot(avg_needed, 'k--', label='Average Card', linewidth=2, alpha=0.5)
        
        if first_place:
            cid = first_place[0]
            plt.plot(card_progress[cid], label=f'Winner (Card {cid})', linewidth=3, color='gold')
            
        if second_place:
            cid = second_place[0]
            plt.plot(card_progress[cid], label=f'2nd (Card {cid})', linewidth=2, color='silver')
            
        # Y axis range depends on card sizes (use maximum squares per card)
        max_squares = max(len(c['events']) for c in cards_data)
        plt.gca().set_ylim(max_squares, 0)
        plt.xlabel("Turn Number")
        plt.ylabel("Squares Needed to Win")
        plt.title("The Race to Bingo", fontsize=20)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        pdf.savefig()
        plt.close()
        
        # --- Fun Stats: The "Almost" List ---
        # Who had 1 square left when the winner won?
        if first_place:
            win_turn = first_place[1]
            # Check progress at this turn (index win_turn)
            almost_winners = []
            for cid, prog in card_progress.items():
                if len(prog) > win_turn:
                    if prog[win_turn] == 1:
                        almost_winners.append(cid)
            
            if almost_winners:
                plt.figure(figsize=(11.69, 8.27))
                plt.text(0.5, 0.8, "Honorable Mentions", ha='center', fontsize=30, weight='bold')
                plt.text(0.5, 0.6, f"Cards with only 1 square left\nwhen the winner won:", ha='center', fontsize=20)
                
                # Wrap text
                txt = ", ".join([f"#{c}" for c in almost_winners])
                wrapped = textwrap.fill(txt, width=40)
                plt.text(0.5, 0.4, wrapped, ha='center', fontsize=16, color='blue')
                
                plt.axis('off')
                pdf.savefig()
                plt.close()

    print(f"Presentation saved to {output_pdf}")

if __name__ == "__main__":
    # --- Configuration ---
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    STATS_PATH = PROJECT_ROOT / "mask" / "segmentation_stats.csv"
    CARDS_PATH = PROJECT_ROOT / "bingo_cards.json"
    IMAGES_DIR = PROJECT_ROOT / "converted_sat_images"
    OUTPUT_PDF = PROJECT_ROOT / "bingo_game_presentation.pdf"
    create_presentation(STATS_PATH, CARDS_PATH, IMAGES_DIR, OUTPUT_PDF)
