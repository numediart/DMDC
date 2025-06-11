import os
import re


index=9
for index in range(1,10):
    listener_dir = './64_frames_windowed_clips/'+str(index)+'_video/listener/'
    speaker_dir = './64_frames_windowed_clips/'+str(index)+'_video/speaker/'
    mfcc_dir = './mfcc_output/'+str(index)+'_video/'

    def extract_base(filename):
        m = re.search(r'(\d+_to_\d+)', filename)
        return m.group(1) if m else None

    def get_basenames(folder):
        basenames = []
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            if not os.path.isfile(path):
                continue
            base = extract_base(filename)
            if base:
                basenames.append(base)
        return basenames

    listener_files = get_basenames(listener_dir)
    speaker_files = get_basenames(speaker_dir)
    mfcc_files = get_basenames(mfcc_dir)

    listener_set = set(listener_files)
    speaker_set = set(speaker_files)
    mfcc_set = set(mfcc_files)

    common = listener_set & speaker_set & mfcc_set

    print(f"Found {len(common)} complete triplets.")

    not_in_common = {
        'listener': listener_set - common,
        'speaker': speaker_set - common,
        'mfcc': mfcc_set - common
    }

    def build_base_to_filename(folder):
        mapping = {}
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            if not os.path.isfile(path):
                continue
            base = extract_base(filename)
            if base:
                mapping[base] = filename
        return mapping

    listener_map = build_base_to_filename(listener_dir)
    speaker_map = build_base_to_filename(speaker_dir)
    mfcc_map = build_base_to_filename(mfcc_dir)

    role_to_dir = {
        'listener': listener_dir,
        'speaker': speaker_dir,
        'mfcc': mfcc_dir
    }
    role_to_map = {
        'listener': listener_map,
        'speaker': speaker_map,
        'mfcc': mfcc_map
    }

    # Show mismatches
    for role, missing in not_in_common.items():
        if missing:
            print(f"\n{role.capitalize()} files not in a complete triplet:")
            for base in missing:
                filename = role_to_map[role].get(base, f"{base} (filename not found)")
                print(f"- {filename}")

    # Ask for deletion after all mismatches are shown
    for role, missing in not_in_common.items():
        if missing:
            resp = input(f"\nDo you want to delete these {role} files? (y/n): ")
            if resp.lower() == 'y':
                for base in missing:
                    filename = role_to_map[role].get(base)
                    if filename:
                        path = os.path.join(role_to_dir[role], filename)
                        try:
                            os.remove(path)
                            print(f"Deleted {path}")
                        except FileNotFoundError:
                            print(f"File not found: {path}")
            else:
                print(f"Kept all {role} files.")
