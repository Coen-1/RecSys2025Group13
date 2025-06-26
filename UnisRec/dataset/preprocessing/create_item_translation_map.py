# create_item_translation_map.py
import json
import os
import argparse
from collections import OrderedDict

def generate_translation_map(meta_data_filepath, session_files_dir, output_map_filepath):
    """
    Generates an item ID translation map.
    It assumes internal IDs found in session files (e.g., "0", "1", "2")
    correspond to the 0-th, 1st, 2nd item key encountered when iterating 
    through the meta_data.json file (maintaining insertion order if Python 3.7+).
    Alternatively, it can try to match based on a small number of shared items if
    some items happen to use the same ID scheme in both.
    """
    print(f"Loading metadata from: {meta_data_filepath}")
    if not os.path.exists(meta_data_filepath):
        print(f"ERROR: Metadata file not found: {meta_data_filepath}")
        return

    # Load metadata preserving order if possible (dict is ordered in Python 3.7+)
    # For older Pythons or to be safe, load into OrderedDict first if a specific order matters
    with open(meta_data_filepath, 'r', encoding='utf-8') as f:
        try:
            # Try loading into OrderedDict to preserve key order from JSON file
            item_metadata_raw = json.load(f, object_pairs_hook=OrderedDict)
            print("Loaded metadata into OrderedDict to preserve key order.")
        except TypeError: # object_pairs_hook not available in older json, or not needed
            print("Loading metadata into standard dict (order might depend on Python version).")
            # Rewind and load as standard dict
            f.seek(0) 
            item_metadata_raw = json.load(f)


    # These are the "original" IDs from your metadata, e.g., "Nxxxxx"
    # The order here is important if we assume index-based mapping.
    meta_data_original_ids_ordered = list(item_metadata_raw.keys())
    print(f"Found {len(meta_data_original_ids_ordered)} unique item keys in metadata.")

    # --- Collect all unique item IDs used in session files (train.json, val.json, test.json) ---
    session_item_ids_set = set()
    interaction_files = ['train.json', 'val.json', 'test.json']

    print("Scanning session files to find all used 'internal' item IDs...")
    for filename in interaction_files:
        filepath = os.path.join(session_files_dir, filename)
        if os.path.exists(filepath):
            print(f"Processing {filepath}...")
            with open(filepath, 'r', encoding='utf-8') as f:
                sessions_raw = json.load(f)
            for session_with_user in sessions_raw:
                if not session_with_user or len(session_with_user) < 2:
                    continue
                # Item IDs are from the second element onwards
                item_ids_in_this_session = [str(item) for item in session_with_user[1:]]
                session_item_ids_set.update(item_ids_in_this_session)
        else:
            print(f"Warning: Session file not found: {filepath}")
            
    print(f"Found {len(session_item_ids_set)} unique item IDs across all session files.")
    if not session_item_ids_set:
        print("ERROR: No item IDs found in session files. Cannot create a map.")
        return

    # --- Attempt to create the mapping ---
    # Strategy: Assume internal IDs like "0", "1", "2" in session files
    # correspond to the 0th, 1st, 2nd item from meta_data_original_ids_ordered.
    translation_map = {}
    unmapped_session_ids = []
    
    # Convert session item IDs that look like integers to actual integers for indexing
    potential_indices = []
    for sid in session_item_ids_set:
        try:
            potential_indices.append(int(sid))
        except ValueError:
            # If an ID from session file is not an integer (e.g., "Nabc"), it can't be an index
            # It might be an original ID already, or a different kind of internal ID.
            # For now, we'll only try to map integer-like session IDs by index.
            print(f"Session item ID '{sid}' is not purely numeric, cannot map by index. Will check for direct match.")
            # If it's an original ID already in metadata, map it to itself.
            if sid in meta_data_original_ids_ordered:
                translation_map[sid] = sid 
            else:
                unmapped_session_ids.append(sid)


    max_potential_index = -1
    if potential_indices:
        max_potential_index = max(potential_indices)

    print(f"Max numeric-like item ID found in session files: {max_potential_index}")
    print(f"Number of items in metadata: {len(meta_data_original_ids_ordered)}")

    if max_potential_index >= len(meta_data_original_ids_ordered):
        print(f"WARNING: Max numeric item ID from sessions ({max_potential_index}) is too large for the number of items in metadata ({len(meta_data_original_ids_ordered)}).")
        print("This suggests the 'index-based mapping' assumption is incorrect or data is inconsistent.")
        print("Mapping will be incomplete for higher IDs.")

    for internal_id_str in session_item_ids_set:
        if internal_id_str in translation_map: # Already mapped (e.g. direct match like "Nxxxx")
            continue
        try:
            internal_id_as_int = int(internal_id_str)
            if 0 <= internal_id_as_int < len(meta_data_original_ids_ordered):
                # Map internal_id_str (e.g., "8") to the 8th original_id from meta_data
                translation_map[internal_id_str] = meta_data_original_ids_ordered[internal_id_as_int]
            else:
                unmapped_session_ids.append(internal_id_str)
        except ValueError:
            # This was already handled, but as a fallback
            if internal_id_str not in unmapped_session_ids:
                 unmapped_session_ids.append(internal_id_str)


    print(f"\nGenerated translation map with {len(translation_map)} entries.")
    if unmapped_session_ids:
        print(f"WARNING: {len(unmapped_session_ids)} item IDs from session files could not be mapped to metadata IDs.")
        print(f"First 10 unmapped session IDs: {unmapped_session_ids[:10]}")
        print("These items will be filtered out during sequence processing if they don't have metadata.")

    with open(output_map_filepath, 'w', encoding='utf-8') as f:
        json.dump(translation_map, f, indent=2)
    print(f"Saved item ID translation map to: {output_map_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an item ID translation map from session internal IDs to metadata original IDs.")
    parser.add_argument('--meta_data_filepath', type=str, required=True, help="Path to the meta_data.json file.")
    parser.add_argument('--session_files_dir', type=str, required=True, help="Directory containing train.json, val.json, test.json.")
    parser.add_argument('--output_map_filepath', type=str, required=True, help="Path to save the generated item_id_internal_to_original.json.")
    
    args = parser.parse_args()
    generate_translation_map(args.meta_data_filepath, args.session_files_dir, args.output_map_filepath)