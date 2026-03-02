import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from collections import defaultdict
import itertools

# --- 1. Authentic Indian Regional Domain Knowledge ---
CITY_CUISINE_MAP = {
    'Mumbai': ['Maharashtrian', 'Street Food', 'Pizza_Chains'],
    'Chandigarh': ['Punjabi', 'North Indian', 'Fast Food'],
    'Bengaluru': ['Karnataka', 'South Indian', 'Pizza_Chains'],
    'Jaipur': ['Rajasthani', 'North Indian', 'Street Food'],
    'Raipur': ['Chhattisgarhi', 'North Indian', 'Fast Food'],
    'Hyderabad': ['Biryani', 'Andhra', 'Fast Food']
}

CUISINE_MENU_MAP = {
    'Punjabi': {
        'main_course': ['Butter Chicken', 'Dal Makhani', 'Sarson Ka Saag', 'Paneer Tikka Masala'],
        'side': ['Garlic Naan', 'Makki Ki Roti', 'Jeera Rice', 'Masala Papad'],
        'drink': ['Sweet Lassi', 'Chaach', 'Coke'],
        'dessert': ['Gajar Ka Halwa', 'Phirni', 'Gulab Jamun']
    },
    'Maharashtrian': {
        'main_course': ['Misal Pav', 'Vada Pav', 'Puran Poli', 'Zunka Bhakar'],
        'side': ['Extra Pav', 'Thecha', 'Kothimbir Vadi', 'Batata Vada'],
        'drink': ['Sol Kadhi', 'Piyush', 'Kokum Sharbat'],
        'dessert': ['Modak', 'Shrikhand']
    },
    'Rajasthani': {
        'main_course': ['Dal Bati Churma', 'Gatte Ki Sabzi', 'Ker Sangri', 'Laal Maas'],
        'side': ['Bajre Ki Roti', 'Mirchi Bada', 'Lehsun Chutney'],
        'drink': ['Jaljeera', 'Masala Chaach'],
        'dessert': ['Malpua', 'Ghevar', 'Moong Dal Halwa']
    },
    'Chhattisgarhi': {
        'main_course': ['Chila', 'Muthia', 'Dubki Kadi', 'Angakar Roti'],
        'side': ['Fara', 'Bafauri', 'Tomato Chutney'],
        'drink': ['Mahua Juice', 'Buttermilk'],
        'dessert': ['Dehrori', 'Khurmi']
    },
    'Karnataka': {
        'main_course': ['Bisi Bele Bath', 'Neer Dosa', 'Ragi Mudde', 'Mangalore Buns'],
        'side': ['Chicken Ghee Roast', 'Coconut Chutney', 'Kosambari'],
        'drink': ['Filter Coffee', 'Badam Milk'],
        'dessert': ['Mysore Pak', 'Obbattu', 'Dharwad Peda']
    },
    'Pizza_Chains': { 
        'main_course': ['Peppy Paneer Pizza', 'Farmhouse Pizza', 'Chicken Dominator'],
        'side': ['Garlic Breadsticks', 'Cheese Dip', 'Stuffed Garlic Bread', 'Chicken Wings'],
        'drink': ['Coke 475ml', 'Fanta', 'Sprite'],
        'dessert': ['Choco Lava Cake', 'Red Velvet Lava']
    },
    'Biryani': {
        'main_course': ['Hyderabadi Chicken Dum Biryani', 'Mutton Biryani', 'Paneer Biryani'],
        'side': ['Mirchi Ka Salan', 'Chicken 65', 'Raita'],
        'drink': ['Thums Up 330ml', 'Sweet Lassi', 'Sprite'],
        'dessert': ['Double Ka Meetha', 'Qubani Ka Meetha']
    }
}

# Fallbacks
for category in ['North Indian', 'South Indian', 'Street Food', 'Fast Food', 'Andhra']:
    if category not in CUISINE_MENU_MAP:
        CUISINE_MENU_MAP[category] = CUISINE_MENU_MAP['Punjabi']

# --- Configurations ---
NUM_USERS = 2000
NUM_RESTAURANTS = 400
NUM_SESSIONS = 8000
TIMES_OF_DAY = ['breakfast', 'lunch', 'evening_snack', 'dinner', 'late_night']

# --- 2. Generate Base Tables ---
print("Generating Users...")
users = pd.DataFrame({
    'user_id': [f'U{str(i).zfill(5)}' for i in range(1, NUM_USERS + 1)],
    'city': np.random.choice(list(CITY_CUISINE_MAP.keys()), NUM_USERS),
    'user_segment': np.random.choice(['budget', 'premium', 'frequent'], NUM_USERS, p=[0.4, 0.2, 0.4]),
    'account_age_days': np.random.randint(1, 1500, NUM_USERS)
})

print("Generating Restaurants...")
rest_data = []
for i in range(1, NUM_RESTAURANTS + 1):
    city = np.random.choice(list(CITY_CUISINE_MAP.keys()))
    cuisine = np.random.choice(CITY_CUISINE_MAP[city])
    is_chain = 1 if cuisine == 'Pizza_Chains' else np.random.choice([0, 1], p=[0.8, 0.2])
    rest_data.append({
        'restaurant_id': f'R{str(i).zfill(4)}',
        'city': city,
        'cuisine_type': cuisine,
        'price_range': np.random.choice(['low', 'medium', 'high'], p=[0.4, 0.4, 0.2]),
        'rating': round(np.random.uniform(3.5, 4.9), 1),
        'is_chain': is_chain
    })
restaurants = pd.DataFrame(rest_data)

print("Generating Menu Items...")
menu_data = []
item_counter = 1
for _, r in restaurants.iterrows():
    cuisine = r['cuisine_type']
    menu_dict = CUISINE_MENU_MAP.get(cuisine, CUISINE_MENU_MAP['Fast Food'])
    
    for category, items in menu_dict.items():
        for item_name in random.sample(items, random.randint(1, len(items))):
            base_price = 250 if category == 'main_course' else 90
            multiplier = 0.7 if r['price_range'] == 'low' else (1.5 if r['price_range'] == 'high' else 1.0)
            
            menu_data.append({
                'item_id': f'M{str(item_counter).zfill(5)}',
                'restaurant_id': r['restaurant_id'],
                'item_name': item_name,
                'category': category,
                'veg_nonveg': 'veg' if 'Paneer' in item_name or 'Dal' in item_name or 'Veg' in item_name else np.random.choice(['veg', 'nonveg']),
                'price': int(base_price * multiplier * random.uniform(0.9, 1.1)),
                'popularity_score': round(random.uniform(0.6, 0.99), 2)
            })
            item_counter += 1
menu_items = pd.DataFrame(menu_data)

print("Generating Rush Patterns...")
rush_data = []
for r_id in restaurants['restaurant_id']:
    for tod in TIMES_OF_DAY:
        rush_mult = round(random.uniform(0.8, 1.2), 2)
        if tod == 'late_night': rush_mult = random.uniform(0.5, 2.0)
        
        rush_data.append({
            'restaurant_id': r_id,
            'time_of_day': tod,
            'rush_multiplier': rush_mult,
            'avg_prep_time_mins': int(15 * rush_mult) + random.randint(5, 10),
            'delivery_partner_availability': round(random.uniform(0.4, 1.0), 2)
        })
restaurant_rush = pd.DataFrame(rush_data)

print("Generating Interactions & Orders...")
interactions, orders, user_history = [], [], []
co_occurrence_dict = defaultdict(int)

for i in range(NUM_SESSIONS):
    user = users.sample(1).iloc[0]
    
    city_rests = restaurants[restaurants['city'] == user['city']]
    if city_rests.empty: continue
    rest = city_rests.sample(1).iloc[0]
    rest_menu = menu_items[menu_items['restaurant_id'] == rest['restaurant_id']]
    
    mains = rest_menu[rest_menu['category'] == 'main_course']
    addons = rest_menu[rest_menu['category'].isin(['side', 'drink', 'dessert'])]
    if mains.empty or addons.empty: continue
    
    main_item = mains.sample(1).iloc[0]
    addon_item = addons.sample(1).iloc[0]
    tod = np.random.choice(TIMES_OF_DAY)
    
    base_prob = addon_item['popularity_score'] - 0.2
    
    if user['user_segment'] == 'budget' and addon_item['price'] > 120: base_prob -= 0.4
    if user['user_segment'] == 'premium': base_prob += 0.2
    
    if 'Pizza' in main_item['item_name'] and ('Lava' in addon_item['item_name'] or 'Garlic' in addon_item['item_name']): base_prob += 0.6
    if 'Biryani' in main_item['item_name'] and 'Salan' in addon_item['item_name']: base_prob += 0.6
    if 'Fara' in main_item['item_name'] or 'Chila' in main_item['item_name']: base_prob += 0.5
    if 'Dal Bati' in main_item['item_name'] and 'Malpua' in addon_item['item_name']: base_prob += 0.5
    
    base_prob = max(0.05, min(0.95, base_prob))
    added = 1 if random.random() < base_prob else 0
    
    session_id = f'S{str(i).zfill(5)}'
    interactions.append({
        'session_id': session_id,
        'user_id': user['user_id'],
        'restaurant_id': rest['restaurant_id'],
        'current_cart_items': f"['{main_item['item_id']}']",
        'candidate_item_id': addon_item['item_id'],
        'candidate_category': addon_item['category'],
        'cart_total_before_add': main_item['price'],
        'time_of_day': tod,
        'added_to_cart': added
    })
    
    final_items = [main_item['item_id']]
    if added: 
        final_items.append(addon_item['item_id'])
        for item1, item2 in itertools.combinations(sorted(final_items), 2):
            co_occurrence_dict[(item1, item2)] += 1
            
    orders.append({
        'order_id': f'O{str(i).zfill(5)}',
        'user_id': user['user_id'],
        'restaurant_id': rest['restaurant_id'],
        'order_timestamp': datetime.now() - timedelta(days=random.randint(0, 60), hours=random.randint(0, 23)),
        'total_order_value': main_item['price'] + (addon_item['price'] if added else 0),
        'items_ordered': str(final_items)
    })
    
    for item_id in final_items:
        user_history.append({
            'user_id': user['user_id'],
            'item_id': item_id,
            'interaction_type': np.random.choice(['view', 'cart_add', 'ordered'], p=[0.2, 0.1, 0.7]),
            'interaction_count': random.randint(1, 10),
            'last_interaction_date': (datetime.now() - timedelta(days=random.randint(1, 100))).strftime('%Y-%m-%d')
        })

df_interactions = pd.DataFrame(interactions)
df_orders = pd.DataFrame(orders)
df_user_history = pd.DataFrame(user_history)

print("Calculating Co-occurrences...")
co_occurrence_data = []
for (item_a, item_b), count in co_occurrence_dict.items():
    co_occurrence_data.append({
        'item_a_id': item_a,
        'item_b_id': item_b,
        'co_purchase_frequency': count,
        'confidence_score': round(min(1.0, count * random.uniform(0.05, 0.15)), 2)
    })
df_co_occurrence = pd.DataFrame(co_occurrence_data)

# --- 4. EXPORT ALL 8 TABLES INTO A SINGLE EXCEL FILE ---
excel_filename = 'Swiggy_Zomato_CSAO_Dataset.xlsx'
print(f"Saving all tables into ONE Excel file: {excel_filename}...")

# Using Pandas ExcelWriter
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    users.to_excel(writer, sheet_name='1_Users', index=False)
    restaurants.to_excel(writer, sheet_name='2_Restaurants', index=False)
    menu_items.to_excel(writer, sheet_name='3_Menu_Items', index=False)
    restaurant_rush.to_excel(writer, sheet_name='4_Rush_Patterns', index=False)
    df_user_history.to_excel(writer, sheet_name='5_User_History', index=False)
    df_co_occurrence.to_excel(writer, sheet_name='6_Co_occurrence', index=False)
    df_orders.to_excel(writer, sheet_name='7_Orders', index=False)
    df_interactions.to_excel(writer, sheet_name='8_Cart_Interactions', index=False)

print("8 tables successfully saved in a single Excel file with separate sheets!")