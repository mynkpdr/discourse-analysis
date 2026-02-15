import json

with open('combined.json') as f:
    data = json.load(f)

items = data['directory_items']
print(f'Total users: {len(items)}')
print(f'Keys per item: {list(items[0].keys())}')
print(f'User keys: {list(items[0]["user"].keys())}')

all_user_keys = set()
all_item_keys = set()
for item in items:
    all_item_keys.update(item.keys())
    all_user_keys.update(item['user'].keys())

print(f'All item keys: {sorted(all_item_keys)}')
print(f'All user keys: {sorted(all_user_keys)}')

trust_levels = {}
for item in items:
    tl = item['user'].get('trust_level', 'N/A')
    trust_levels[tl] = trust_levels.get(tl, 0) + 1
print(f'Trust levels: {trust_levels}')

groups = {}
for item in items:
    g = item['user'].get('primary_group_name', 'None')
    groups[g] = groups.get(g, 0) + 1
print(f'Groups: {groups}')
