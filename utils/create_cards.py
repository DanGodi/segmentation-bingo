import polars as pl
import numpy as np
import random
from pathlib import Path
from tqdm.notebook import tqdm
import ipywidgets as widgets
from IPython.display import display
import json
import concurrent.futures
import multiprocessing

def generate_events(matrix, feature_event_types=None):
    """
    Generates candidate events from the counts matrix using Polars.
    Returns:
        events_df: Polars DataFrame with metadata (no masks).
        truth_matrix: Numpy boolean array (n_images, n_events).
    """
    events_meta = []
    masks = []
    
    features = [c for c in matrix.columns if c != "image"]
    
    for feat in features:
        col = matrix[feat]
        
        if feature_event_types is not None:
            selected_types = feature_event_types.get(feat, [])
        else:
            selected_types = ['exists', 'threshold', 'exact']

        col_np = col.to_numpy()
        max_val = col_np.max()
        
        # 1. Existence
        if 'exists' in selected_types:
            mask = col_np > 0
            prob = mask.mean()
            if 0.001 <= prob <= 0.95:
                events_meta.append({
                    "description": f"Contains {feat}",
                    "type": "exists",
                    "feature": feat,
                    "probability": prob
                })
                masks.append(mask)
        
        # 2. Thresholds
        if 'threshold' in selected_types and max_val > 1:
            for n in range(1, min(int(max_val), 6)):
                mask = col_np > n
                prob = mask.mean()
                if 0.05 <= prob <= 0.95:
                    events_meta.append({
                        "description": f"More than {n} {feat}s",
                        "type": "threshold",
                        "feature": feat,
                        "probability": prob
                    })
                    masks.append(mask)

        # 3. Exact Counts
        if 'exact' in selected_types and max_val >= 1:
            for n in range(1, min(int(max_val) + 1, 6)):
                mask = col_np == n
                prob = mask.mean()
                if 0.05 <= prob <= 0.95:
                    events_meta.append({
                        "description": f"Exactly {n} {feat}{'s' if n>1 else ''}",
                        "type": "exact",
                        "feature": feat,
                        "probability": prob
                    })
                    masks.append(mask)
    
    if not events_meta:
        return pl.DataFrame(), np.array([])

    return pl.DataFrame(events_meta), np.stack(masks).T

def calculate_turns_to_win_vectorized(card_indices_list, truth_matrix, n_simulations=1000):
    """
    Calculates average turns to win for a LIST of cards in parallel (vectorized numpy).
    card_indices_list: List of lists, where each inner list is indices of events for a card.
    truth_matrix: (n_images, n_total_events) boolean numpy array.
    """
    n_images, _ = truth_matrix.shape
    n_cards = len(card_indices_list)
    
    # Pre-select columns for all cards
    card_masks = [truth_matrix[:, idxs] for idxs in card_indices_list]
    card_masks_array = np.stack(card_masks, axis=0) # (n_cards, n_images, card_size)
    
    turns_needed = np.zeros((n_simulations, n_cards), dtype=int)
    deck = np.arange(n_images)
    
    for s in range(n_simulations):
        np.random.shuffle(deck)
        shuffled = card_masks_array[:, deck, :]
        
        # Cumulative coverage along image axis
        covered_cum = np.maximum.accumulate(shuffled, axis=1)
        
        # Check if all events in card are covered
        all_covered = covered_cum.all(axis=2)
        
        has_win = all_covered.any(axis=1)
        wins = np.argmax(all_covered, axis=1) + 1
        wins[~has_win] = n_images
        
        turns_needed[s, :] = wins
        
    return turns_needed.mean(axis=0)

def find_valid_card(args):
    """
    Worker function to find a single valid card configuration.
    """
    unique_features, feature_groups, truth_matrix, card_size, target_difficulty, tolerance, max_attempts = args
    
    for _ in range(max_attempts):
        selected_features = random.sample(unique_features, card_size)
        card_idxs = []
        for feat in selected_features:
            possible_idxs = feature_groups[feat]
            card_idxs.append(random.choice(possible_idxs))
            
        # Quick check
        est_diff = calculate_turns_to_win_vectorized([card_idxs], truth_matrix, n_simulations=50)[0]
        
        if abs(est_diff - target_difficulty) < tolerance * 2:
            # Precise check
            precise_diff = calculate_turns_to_win_vectorized([card_idxs], truth_matrix, n_simulations=5000)[0]
            if abs(precise_diff - target_difficulty) < tolerance:
                return card_idxs, precise_diff
                
    return None, None

def run_tournament_simulation(cards, truth_matrix, n_simulations):
    """
    Simulates a game with all cards playing against each other.
    Returns win counts for each card.
    """
    n_images = truth_matrix.shape[0]
    n_cards = len(cards)
    
    card_masks = np.stack([truth_matrix[:, c] for c in cards], axis=0)
    win_counts = np.zeros(n_cards)
    
    batch_size = 1000
    n_batches = (n_simulations + batch_size - 1) // batch_size
    deck = np.arange(n_images)
    
    for _ in range(n_batches):
        current_batch = min(batch_size, n_simulations)
        n_simulations -= current_batch
        
        batch_wins = np.zeros((current_batch, n_cards), dtype=int)
        
        for b in range(current_batch):
            np.random.shuffle(deck)
            shuffled = card_masks[:, deck, :]
            
            # Check coverage
            covered = np.maximum.accumulate(shuffled, axis=1).all(axis=2)
            
            turns = np.argmax(covered, axis=1)
            did_win = covered.any(axis=1)
            turns[~did_win] = n_images + 9999
            
            batch_wins[b, :] = turns
            
        # Determine winners
        min_turns = batch_wins.min(axis=1)
        is_winner = (batch_wins == min_turns[:, None])
        
        # Split points for ties
        n_winners = is_winner.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            points = is_winner.astype(float) / n_winners[:, None]
            points[~np.isfinite(points)] = 0
            
        win_counts += points.sum(axis=0)
        
    return win_counts

def create_cards_interactive(stats_path, output_path, num_cards=50, card_size=10, tolerance=1, target_difficulty=None):
    stats_path = Path(stats_path)
    output_path = Path(output_path)
    
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    # --- Polars Load ---
    df = pl.read_csv(stats_path)
    df = df.with_columns(pl.col("n_objects").fill_null(0).cast(pl.Int32))
    
    # Pivot
    counts_matrix = df.pivot(index="image", columns="feature", values="n_objects", aggregate_function="first").fill_null(0)
    features = [c for c in counts_matrix.columns if c != "image"]

    # --- UI Setup ---
    feature_widgets = {}
    for feat in features:
        exists_cb = widgets.Checkbox(value=True, description='Contains')
        threshold_cb = widgets.Checkbox(value=False, description='More than N')
        exact_cb = widgets.Checkbox(value=False, description='Exactly N')
        box = widgets.VBox([widgets.Label(f"Feature: {feat}"), exists_cb, threshold_cb, exact_cb])
        feature_widgets[feat] = box

    widgets_list = [feature_widgets[feat] for feat in features]
    accordion = widgets.Accordion(children=widgets_list, titles=features)
    display(accordion)

    confirm_button = widgets.Button(description="Confirm Selections & Generate Cards")
    output = widgets.Output()

    def on_confirm_clicked(b):
        with output:
            output.clear_output()
            feature_event_types = {}
            for feat, box in feature_widgets.items():
                exists_val = box.children[1].value
                threshold_val = box.children[2].value
                exact_val = box.children[3].value
                selected = []
                if exists_val: selected.append('exists')
                if threshold_val: selected.append('threshold')
                if exact_val: selected.append('exact')
                feature_event_types[feat] = selected
            
            print("Generating candidate events (Polars)...")
            events_df, truth_matrix = generate_events(counts_matrix, feature_event_types)
            print(f"Generated {len(events_df)} candidate events.")
            
            # Group features
            events_df = events_df.with_row_index("idx")
            feature_groups = {}
            for name, data in events_df.group_by("feature"):
                feature_groups[name] = data["idx"].to_list()
            
            unique_features = list(feature_groups.keys())

            if len(unique_features) < card_size:
                print(f"Error: Not enough unique features ({len(unique_features)}) for card size {card_size}.")
                return

            # --- Target Difficulty ---
            nonlocal target_difficulty
            if target_difficulty is None:
                print("Estimating baseline difficulty...")
                sample_cards = []
                for _ in range(50):
                    feats = random.sample(unique_features, card_size)
                    idxs = [random.choice(feature_groups[f]) for f in feats]
                    sample_cards.append(idxs)
                
                diffs = calculate_turns_to_win_vectorized(sample_cards, truth_matrix, n_simulations=100)
                valid_diffs = diffs[diffs < truth_matrix.shape[0]]
                
                if len(valid_diffs) > 0:
                    target_difficulty = float(np.median(valid_diffs))
                    print(f"Calculated Target Average Turns to Win (Median): {target_difficulty:.2f}")
                else:
                    print("Could not find valid cards for difficulty estimation.")
                    return
            else:
                print(f"Using Manual Target: {target_difficulty:.2f}")

            # --- Initial Generation (Parallel) ---
            print(f"Generating {num_cards} initial cards...")
            
            final_cards = []
            final_stats = []
            
            worker_args = (unique_features, feature_groups, truth_matrix, card_size, target_difficulty, tolerance, 5000)
            
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(find_valid_card, worker_args) for _ in range(num_cards)]
                for f in tqdm(concurrent.futures.as_completed(futures), total=num_cards, desc="Initial Gen"):
                    card, stat = f.result()
                    if card:
                        final_cards.append(card)
                        final_stats.append(stat)

            if len(final_cards) < num_cards:
                print(f"Warning: Only generated {len(final_cards)} valid cards initially.")
            
            # --- Balancing Loop ---
            print("Starting Balancing Loop (Win Percentage)...")
            
            max_iterations = 50
            
            for iteration in range(max_iterations):
                current_num = len(final_cards)
                if current_num == 0: break
                
                target_win_rate = 1.0 / current_num
                win_rate_tolerance = 0.35 * target_win_rate
                
                # Run Tournament
                total_sims = 100000
                n_cores = multiprocessing.cpu_count()
                sims_per_core = total_sims // n_cores
                
                print(f"  Iter {iteration+1}: Running {total_sims} tournament simulations...")
                
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futs = [executor.submit(run_tournament_simulation, final_cards, truth_matrix, sims_per_core) for _ in range(n_cores)]
                    results = [f.result() for f in futs]
                
                total_wins = np.sum(results, axis=0)
                win_rates = total_wins / total_sims
                
                deviations = np.abs(win_rates - target_win_rate)
                bad_indices = np.where(deviations > win_rate_tolerance)[0]
                
                if len(bad_indices) == 0:
                    print("  Converged! All cards within win rate tolerance.")
                    break
                
                print(f"  Found {len(bad_indices)} cards outside tolerance ({win_rate_tolerance:.4f}).")
                
                # Keep good cards
                good_indices = [i for i in range(current_num) if i not in bad_indices]
                new_final_cards = [final_cards[i] for i in good_indices]
                new_final_stats = [final_stats[i] for i in good_indices]
                
                # Regenerate missing
                needed = num_cards - len(new_final_cards)
                print(f"  Regenerating {needed} cards...")
                
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futures = [executor.submit(find_valid_card, worker_args) for _ in range(needed)]
                    for f in tqdm(concurrent.futures.as_completed(futures), total=needed, desc="Regen", leave=False):
                        card, stat = f.result()
                        if card:
                            new_final_cards.append(card)
                            new_final_stats.append(stat)
                
                final_cards = new_final_cards
                final_stats = new_final_stats

            # --- Export ---
            cards_data = []
            descriptions = events_df["description"].to_list()
            
            for i, idxs in enumerate(final_cards):
                card_events = [descriptions[idx] for idx in idxs]
                cards_data.append({
                    "card_id": i + 1,
                    "events": card_events,
                    "avg_turns_to_win_isolation": final_stats[i]
                })
                
            with open(output_path, "w") as f:
                json.dump(cards_data, f, indent=2)
                
            print(f"Saved {len(cards_data)} cards to {output_path}")

    confirm_button.on_click(on_confirm_clicked)
    display(confirm_button, output)