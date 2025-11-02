#!/usr/bin/env python3

import os
import argparse
import shutil

def main():
    parser = argparse.ArgumentParser(
        description='Delete folders under oat-output except specified ones'
    )
    parser.add_argument(
        '--keep',
        nargs='+',
        required=True,
        help='List of folder names to keep (relative to oat-output)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    
    args = parser.parse_args()
    
    base_dir = '/ephemeral/games-workspace/spiral_new/oat-output'
    keep_folders = set(args.keep)
    
    print(f'Base directory: {base_dir}')
    print(f'Folders to keep: {keep_folders}')
    print(f'Dry run: {args.dry_run}')
    print('-' * 60)
    
    all_items = os.listdir(base_dir)
    
    for item in all_items:
        item_path = os.path.join(base_dir, item)
        
        if os.path.isdir(item_path):
            if item not in keep_folders:
                if args.dry_run:
                    print(f'[DRY RUN] Would delete: {item_path}')
                else:
                    print(f'Deleting: {item_path}')
                    shutil.rmtree(item_path)
            else:
                print(f'Keeping: {item_path}')

if __name__ == '__main__':
    main()

