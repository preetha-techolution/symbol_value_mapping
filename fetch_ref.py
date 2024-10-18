from spellchecker import SpellChecker
from thefuzz import fuzz  
import time 

spell = SpellChecker()  # loads default word frequency list

spell.distance = 5

global REFERENCE_NUMBERS

def load_reference_numbers(text_file_path):
    global REFERENCE_NUMBERS
    with open(text_file_path, 'r') as file:
        REFERENCE_NUMBERS = file.read().splitlines()
    spell.word_frequency.load_text_file(text_file_path)


load_reference_numbers("reference_numbers.txt")

def fallback(reference_number):
    global REFERENCE_NUMBERS
    best_reference_number = None 
    best_score = 0
    for ref_no in REFERENCE_NUMBERS:
        ratio = fuzz.ratio(reference_number, ref_no)
        if ratio > best_score:
            best_score = ratio
            best_reference_number = ref_no

    return best_reference_number


def get_product_suggestions(possible_reference_numbers, scores):
    
    # Combine the lists using zip
    combined = zip(scores, possible_reference_numbers)

    # Sort the combined list by scores in descending order
    sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)

    # Unzip the sorted pairs back into two lists
    sorted_scores, sorted_strings = zip(*sorted_combined)

    # Convert to list (since zip returns tuples)
    sorted_scores = list(sorted_scores)
    sorted_strings = list(sorted_strings)

    return sorted_strings[:3] if len(sorted_strings)>3 else sorted_strings


def get_nearest_ref_number(reference_number):
    try:
        if reference_number == "":
            return None
        
        s = time.time()
        best_score = 0
        threshold = 0.85
        best_reference_number = None 


        possible_ref_no = spell.candidates(reference_number)
        
        print(f'Possible reference numbers : {possible_ref_no}')

        
        if possible_ref_no is None or type(possible_ref_no)==set:
            print('Fallback')
            reference_number = fallback(reference_number)
            return reference_number
        
        scores = []
    
        for ref_no in possible_ref_no:
            score = fuzz.ratio(reference_number, ref_no)
            scores.append(score)
            
            if score > threshold and score>best_score:
                best_reference_number = ref_no 
                best_score = score 

        suggestions = get_product_suggestions(possible_ref_no, scores)
        print(suggestions)
        e = time.time()
        print(f'{e-s} for seraching one reference number')
        return best_reference_number.upper()
    except Exception as e:
        print(f'{e} at line number 29, in refno_search.py')