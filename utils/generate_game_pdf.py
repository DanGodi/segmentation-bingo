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

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATS_PATH = PROJECT_ROOT / "mask" / "segmentation_stats.csv"
CARDS_PATH = PROJECT_ROOT / "bingo_cards.json"
IMAGES_DIR = PROJECT_ROOT / "converted_sat_images"
OUTPUT_PDF = PROJECT_ROOT / "bingo_game_presentation.pdf"

# --- Helper Functions ---

def load_data():
    """Loads stats and cards data."""
    if not STATS_PATH.exists():
        raise FileNotFoundError(f"Stats file not found: {STATS_PATH}")
    if not CARDS_PATH.exists():
        raise FileNotFoundError(f"Cards file not found: {CARDS_PATH}")
        
    df = pd.read_csv(STATS_PATH)
    df["n_objects"] = df["n_objects"].fillna(0).astype(int)
    
    # Pivot to get Image x Feature matrix
    counts_matrix = df.pivot_table(index="image", columns="feature", values="n_objects", fill_value=0)
    
    with open(CARDS_PATH, "r") as f:
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
    # Note: The generator added 's' at the end. We need to be careful matching the feature name.
    # We'll try to match the feature name from the known columns.
    if event_str.startswith("More than "):
        match = re.match(r"More than (\d+) (.+)", event_str)
        if match:
            n = int(match.group(1))
            rest = match.group(2)
            # The generator added 's' to the end. Let's try removing it.
            # But what if the feature itself ends in s?
            # Let's iterate through known features to find the match.
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
    # Initialize card state (set of unfulfilled events)
    # Actually, simpler: keep track of how many events are satisfied
    
    # Pre-calculate requirements for each card
    # card_requirements[card_id] = [ (event_str, is_satisfied_bool) ]
    # But is_satisfied depends on the image.
    # Bingo rule: A square is marked if the CURRENT image satisfies the condition.
    # Wait, standard Bingo: You mark the square if the CALLED item matches.
    # In this game: "Show an image". Does the image satisfy "Contains Pool"?
    # If yes, you mark that square.
    # Once marked, it stays marked.
    
    n_cards = len(cards_data)
    card_progress = {c['card_id']: [10] for c in cards_data} # Starts with 10 needed
    card_events = {c['card_id']: c['events'] for c in cards_data}
    card_status = {c['card_id']: [False]*10 for c in cards_data} # 10 False values
    
    winners = []
    
    for turn_idx, img_name in enumerate(image_order):
        # Get features for this image
        if img_name in counts_matrix.index:
            row = counts_matrix.loc[img_name]
            
            # Update all cards
            for c in cards_data:
                cid = c['card_id']
                if cid in [w[0] for w in winners]:
                    card_progress[cid].append(0)
                    continue # Already won
                
                # Check each event
                for i, event in enumerate(card_events[cid]):
                    if not card_status[cid][i]: # If not yet marked
                        if parse_event(event, row):
                            card_status[cid][i] = True
                            
                # Record progress
                needed = 10 - sum(card_status[cid])
                card_progress[cid].append(needed)
                
                if needed == 0:
                    winners.append((cid, turn_idx + 1))
        else:
            # Image not in stats? Just record progress as same as last
            for cid in card_progress:
                card_progress[cid].append(card_progress[cid][-1])
                
    return winners, card_progress

def create_presentation():
    print("Generating Bingo Presentation...")
    
    # 1. Load Data
    counts_matrix, cards_data = load_data()
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
    
    print(f"The winner won on turn {first_place[1]}")
    
    # 4. Generate PDF
    with PdfPages(OUTPUT_PDF) as pdf:
        
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
        # User said "randomly decides an order... and displays them".
        # Let's show all.
        
        for i, img_name in enumerate(all_images):
            img_path = IMAGES_DIR / img_name
            
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
                plt.text(0.5, 0.5, f"Image not found:\n{img_name}", ha='center')
                plt.axis('off')
                
            # Footer
            plt.text(0.5, 0.02, f"Image ID: {img_name}", ha='center', fontsize=8, color='gray', transform=plt.gcf().transFigure)
            
            pdf.savefig()
            plt.close()
            
            # Optimization: If we have a winner, maybe we mark it? 
            # No, that spoils the fun. The PDF is the deck.
            
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
                # Text
                plt.text(x[idx], heights[idx] + 0.1, f"Card #{card_id}", ha='center', fontsize=20, weight='bold')
                plt.text(x[idx], heights[idx] - 0.5, f"Won on\nTurn {turn}", ha='center', fontsize=14, color='white')
                plt.text(x[idx], 0.2, labels[idx], ha='center', fontsize=16, weight='bold')
                
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
        all_progress = np.array([card_progress[c['card_id']] for c in cards_data])
        # Pad with last value if lengths differ (shouldn't happen if we simulated full game)
        avg_needed = np.mean(all_progress, axis=0)
        
        plt.plot(avg_needed, 'k--', label='Average Card', linewidth=2, alpha=0.5)
        
        if first_place:
            pid = first_place[0]
            plt.plot(card_progress[pid], 'r-', label=f'Winner (Card {pid})', linewidth=3)
            
        if second_place:
            pid = second_place[0]
            plt.plot(card_progress[pid], 'b-', label=f'2nd (Card {pid})', linewidth=2)
            
        plt.gca().invert_yaxis() # 10 down to 0
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
            almost_winners = []
            for cid, prog in card_progress.items():
                if cid == first_place[0]: continue
                if prog[win_turn] == 1:
                    almost_winners.append(cid)
            
            plt.figure(figsize=(11.69, 8.27))
            plt.axis('off')
            plt.text(0.5, 0.9, "The 'So Close!' Award", ha='center', fontsize=30, weight='bold')
            plt.text(0.5, 0.8, f"(Cards with only 1 square left when Card #{first_place[0]} won)", ha='center', fontsize=16)
            
            if almost_winners:
                txt = ", ".join([f"#{cid}" for cid in almost_winners])
                wrapped_txt = "\n".join(textwrap.wrap(txt, width=40))
                plt.text(0.5, 0.5, wrapped_txt, ha='center', va='center', fontsize=24, color='blue')
            else:
                plt.text(0.5, 0.5, "No one else was close!", ha='center', fontsize=24)
                
            pdf.savefig()
            plt.close()

    print(f"Presentation saved to {OUTPUT_PDF}")

if __name__ == "__main__":
    create_presentation()
