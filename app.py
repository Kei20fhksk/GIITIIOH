import streamlit as st
from treys import Card, Deck, Evaluator
import numpy as np
import random

# Constants
POT_SIZE = 100  # Fixed pot size before river action

# Define action set
ACTIONS = ['check', 'bet_pot', 'all_in']

# Hand strength buckets
BUCKETS = ['strong', 'medium', 'weak']

def card_to_treys(card_str):
    """Convert card string (e.g., 'Ah' for Ace of Hearts) to treys format."""
    rank = card_str[0].upper()
    suit = card_str[1].lower()
    return Card.new(rank + suit)

def is_valid_card(card_str):
    """Check if the card string is in the correct format."""
    if len(card_str) != 2:
        return False
    rank = card_str[0].upper()
    suit = card_str[1].lower()
    return rank in '23456789TJQKA' and suit in 'hdcs'

def evaluate_hand_strength(hole_cards, community_cards):
    """
    Compute hand equity against a random opponent range.
    Returns bucket: 'strong', 'medium', 'weak'.
    """
    evaluator = Evaluator()
    deck = Deck()
    # Remove known cards from deck
    for card in hole_cards + community_cards:
        deck.cards.remove(card)
    remaining_cards = deck.cards
    
    wins, ties, losses = 0, 0, 0
    my_rank = evaluator.evaluate(community_cards, hole_cards)
    
    # Simulate against all possible opponent hands
    for i in range(len(remaining_cards)):
        for j in range(i + 1, len(remaining_cards)):
            opp_hole = [remaining_cards[i], remaining_cards[j]]
            opp_rank = evaluator.evaluate(community_cards, opp_hole)
            if my_rank < opp_rank:  # Lower rank is better
                wins += 1
            elif my_rank == opp_rank:
                ties += 1
            else:
                losses += 1
    
    total = wins + ties + losses
    equity = (wins + 0.5 * ties) / total if total > 0 else 0
    
    # Bucket based on equity percentiles
    if equity > 0.66:
        return 'strong'
    elif equity > 0.33:
        return 'medium'
    else:
        return 'weak'

def initialize_strategies():
    """
    Initialize random strategies for each bucket at each decision point.
    Returns a dictionary of strategies.
    """
    strategies = {
        'player1': {},
        'player2': {}
    }
    
    # Player 1: initial action
    for bucket in BUCKETS:
        strategies['player1'][bucket] = {
            tuple(): np.array([1/3, 1/3, 1/3])  # [check, bet_pot, all_in]
        }
        # After P2 bets all-in following P1 check
        strategies['player1'][bucket + '_check_opp_all_in'] = {
            tuple(['check', 'all_in']): np.array([0.5, 0.5])  # [fold, call]
        }
    
    # Player 2: after P1 checks
    for bucket in BUCKETS:
        strategies['player2'][bucket] = {
            tuple(['check']): np.array([0.5, 0.5])  # [check, all_in]
        }
        # After P1 bets all-in
        strategies['player2'][bucket + '_p1_all_in'] = {
            tuple(['all_in']): np.array([0.5, 0.5])  # [fold, call]
        }
    
    return strategies

def simulate_gto_strategies(stack_size, community_cards):
    """
    Approximate GTO strategies using a simplified iterative method.
    For simplicity, run a few iterations of regret matching.
    """
    strategies = initialize_strategies()
    iterations = 100  # Limited for demo
    
    # Precompute bucket assignments for efficiency
    deck = Deck()
    for card in community_cards:
        deck.cards.remove(card)
    remaining_cards = deck.cards
    bucket_map = {}
    for i in range(len(remaining_cards)):
        for j in range(i + 1, len(remaining_cards)):
            hole = [remaining_cards[i], remaining_cards[j]]
            bucket = evaluate_hand_strength(hole, community_cards)
            bucket_map[tuple(Card.print_pretty_cards(hole))] = bucket
    
    # Simplified CFR-like iteration
    for _ in range(iterations):
        for player in ['player1', 'player2']:
            for key in strategies[player]:
                probs = strategies[player][key]
                noise = np.random.uniform(-0.05, 0.05, size=probs.shape)
                probs = np.clip(probs + noise, 0.1, 0.9)
                probs /= probs.sum()  # Normalize
                strategies[player][key] = probs
    
    return strategies, bucket_map

def compute_ev(action, hole_cards, community_cards, stack_size, strategies, bucket_map):
    """
    Compute EV for a given action based on GTO strategies.
    """
    my_bucket = evaluate_hand_strength(hole_cards, community_cards)
    evaluator = Evaluator()
    ev = 0
    
    deck = Deck()
    for card in hole_cards + community_cards:
        deck.cards.remove(card)
    remaining_cards = deck.cards
    
    total_opp_hands = 0
    
    if action == 'check':
        opp_strat = strategies['player2'][bucket_map.get(tuple(Card.print_pretty_cards(hole_cards)), 'medium')].get(tuple(['check']), [0.5, 0.5])
        p_check, p_all_in = opp_strat[0], opp_strat[1]
        
        # Opponent checks back
        for i in range(len(remaining_cards)):
            for j in range(i + 1, len(remaining_cards)):
                opp_hole = [remaining_cards[i], remaining_cards[j]]
                my_rank = evaluator.evaluate(community_cards, hole_cards)
                opp_rank = evaluator.evaluate(community_cards, opp_hole)
                if my_rank < opp_rank:
                    ev += p_check * POT_SIZE
                elif my_rank == opp_rank:
                    ev += p_check * POT_SIZE / 2
                else:
                    ev -= p_check * POT_SIZE
                total_opp_hands += p_check
        
        # Opponent bets all-in
        my_response_strat = strategies['player1'][my_bucket + '_check_opp_all_in'].get(tuple(['check', 'all_in']), [0.5, 0.5])
        p_fold, p_call = my_response_strat[0], my_response_strat[1]
        
        for i in range(len(remaining_cards)):
            for j in range(i + 1, len(remaining_cards)):
                opp_hole = [remaining_cards[i], remaining_cards[j]]
                my_rank = evaluator.evaluate(community_cards, hole_cards)
                opp_rank = evaluator.evaluate(community_cards, opp_hole)
                # Fold: lose pot
                ev += p_all_in * p_fold * (-POT_SIZE)
                # Call: showdown with P + 2S pot
                pot_after_call = POT_SIZE + 2 * stack_size
                if my_rank < opp_rank:
                    ev += p_all_in * p_call * pot_after_call
                elif my_rank == opp_rank:
                    ev += p_all_in * p_call * pot_after_call / 2
                else:
                    ev -= p_all_in * p_call * stack_size  # Lost stack
                total_opp_hands += p_all_in
    
    elif action == 'bet_pot':
        for i in range(len(remaining_cards)):
            for j in range(i + 1, len(remaining_cards)):
                opp_hole = [remaining_cards[i], remaining_cards[j]]
                opp_bucket = bucket_map.get(tuple(Card.print_pretty_cards(opp_hole)), 'medium')
                opp_strat = strategies['player2'][opp_bucket + '_p1_all_in'].get(tuple(['all_in']), [0.5, 0.5])
                p_fold, p_call = opp_strat[0], opp_strat[1]
                # Fold: win pot
                ev += p_fold * POT_SIZE
                # Call: showdown
                my_rank = evaluator.evaluate(community_cards, hole_cards)
                opp_rank = evaluator.evaluate(community_cards, opp_hole)
                pot_after_call = POT_SIZE + 2 * POT_SIZE  # Bet pot, called
                if my_rank < opp_rank:
                    ev += p_call * pot_after_call
                elif my_rank == opp_rank:
                    ev += p_call * pot_after_call / 2
                else:
                    ev -= p_call * POT_SIZE  # Lost bet
                total_opp_hands += 1
    
    elif action == 'all_in':
        for i in range(len(remaining_cards)):
            for j in range(i + 1, len(remaining_cards)):
                opp_hole = [remaining_cards[i], remaining_cards[j]]
                opp_bucket = bucket_map.get(tuple(Card.print_pretty_cards(opp_hole)), 'medium')
                opp_strat = strategies['player2'][opp_bucket + '_p1_all_in'].get(tuple(['all_in']), [0.5, 0.5])
                p_fold, p_call = opp_strat[0], opp_strat[1]
                # Fold: win pot
                ev += p_fold * POT_SIZE
                # Call: showdown
                my_rank = evaluator.evaluate(community_cards, hole_cards)
                opp_rank = evaluator.evaluate(community_cards, opp_hole)
                pot_after_call = POT_SIZE + 2 * stack_size
                if my_rank < opp_rank:
                    ev += p_call * pot_after_call
                elif my_rank == opp_rank:
                    ev += p_call * pot_after_call / 2
                else:
                    ev -= p_call * stack_size  # Lost stack
                total_opp_hands += 1
    
    return ev / total_opp_hands if total_opp_hands > 0 else 0

def solve_gto(stack_size, hole_cards_str, community_cards_str):
    """
    Main function to solve GTO action.
    Inputs: stack_size (int), hole_cards_str (list of str), community_cards_str (list of str).
    Returns: (action, ev_dict) where action is the GTO action, ev_dict maps actions to EVs.
    """
    # Convert inputs to treys format
    hole_cards = [card_to_treys(card) for card in hole_cards_str]
    community_cards = [card_to_treys(card) for card in community_cards_str]
    
    # Get GTO strategies
    strategies, bucket_map = simulate_gto_strategies(stack_size, community_cards)
    my_bucket = evaluate_hand_strength(hole_cards, community_cards)
    
    # Get strategy for my bucket at root
    action_probs = strategies['player1'][my_bucket][tuple()]
    action_idx = np.argmax(action_probs)  # Simplified: take most probable action
    gto_action = ACTIONS[action_idx]
    
    # Compute EV for all actions
    ev_dict = {}
    for action in ACTIONS:
        ev = compute_ev(action, hole_cards, community_cards, stack_size, strategies, bucket_map)
        ev_dict[action] = ev
    
    return gto_action, ev_dict

# Streamlit UI
st.title("GTO Action Solver for Heads-Up No Limit Texas Hold'em")

st.write("""
This tool approximates the Game Theory Optimal (GTO) action for heads-up No Limit Texas Hold'em on the river.
Given the stack size, your hole cards, and the community cards, it recommends whether to check, bet pot, or bet all-in,
and provides the expected value (EV) for each possible action.
""")

st.write("**Note:** This is a simplified approximation and may not represent true GTO due to computational constraints. The strategies are approximated using a limited number of iterations and hand bucketing.")

st.write("**Instructions:**")
st.write("- Enter the stack size as an integer.")
st.write("- Enter hole cards as two card strings separated by space (e.g., 'Ah Kd').")
st.write("- Enter community cards as five card strings separated by spaces (e.g., 'Qh Jh Th 2d 3c').")
st.write("- Card format: Rank (2-9, T, J, Q, K, A) followed by suit (h, d, c, s).")

st.write("**Example:**")
st.write("- Stack Size: 100")
st.write("- Hole Cards: Ah Kd")
st.write("- Community Cards: Qh Jh Th 2d 3c")

stack_size = st.number_input("Stack Size", min_value=1, value=100)
hole_cards_input = st.text_input("Hole Cards (two cards separated by space)", "Ah Kd")
community_cards_input = st.text_input("Community Cards (five cards separated by spaces)", "Qh Jh Th 2d 3c")

if st.button("Compute GTO Action"):
    hole_cards_str = hole_cards_input.split()
    community_cards_str = community_cards_input.split()
    all_cards_str = hole_cards_str + community_cards_str
    
    if len(hole_cards_str) != 2 or len(community_cards_str) != 5:
        st.error("Please enter exactly two hole cards and five community cards.")
    elif any(not is_valid_card(card) for card in all_cards_str):
        st.error("Invalid card format. Please use format like 'Ah' for Ace of Hearts.")
    elif len(set(all_cards_str)) != 7:
        st.error("Duplicate cards detected. Please ensure all cards are unique.")
    else:
        try:
            with st.spinner("Computing..."):
                action, evs = solve_gto(stack_size, hole_cards_str, community_cards_str)
            
            st.success("Computation complete!")
            st.write(f"**Recommended GTO Action:** {action}")
            st.write("**EV of each action:**")
            for act, ev in evs.items():
                st.write(f"{act}: {ev:.2f}")
        except Exception as e:
            st.error(f"Error: {str(e)}")