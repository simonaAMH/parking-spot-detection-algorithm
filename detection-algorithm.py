import cv2
import numpy as np
import json
from sklearn.cluster import DBSCAN

WINDOW_WIDTH = 480
WINDOW_HEIGHT = 320

def detect_parking_spots_with_occupancy(empty_lot_path, filled_lot_path, output_json_path, output_image_path):
    empty_img = cv2.imread(empty_lot_path)
    if empty_img is None:
        print(f"Error: Could not load empty lot image from {empty_lot_path}")
        return []

    filled_img = cv2.imread(filled_lot_path)
    if filled_img is None:
        print(f"Error: Could not load filled lot image from {filled_lot_path}")
        return []

    windows = [
        '1. Empty Lot', '2. White Line Mask', '3. Line Segments', 
        '4. Merged & Extended Grid', '5. Spots with Occupancy Status'
    ]
    for i, name in enumerate(windows):
        cv2.namedWindow(name)
        cv2.moveWindow(name, (i % 3) * (WINDOW_WIDTH + 10), (i // 3) * (WINDOW_HEIGHT + 40))

    step1_img = empty_img.copy()
    cv2.imshow('1. Empty Lot Reference', cv2.resize(step1_img, (WINDOW_WIDTH, WINDOW_HEIGHT)))

    white_mask = detect_white_lines(empty_img)
    step2_img = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
    cv2.imshow('2. White Line Mask', cv2.resize(step2_img, (WINDOW_WIDTH, WINDOW_HEIGHT)))

    lines = detect_line_segments(white_mask)
    step3_img = empty_img.copy()
    for x1, y1, x2, y2 in lines['horizontal']:
        cv2.line(step3_img, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow
    for x1, y1, x2, y2 in lines['vertical']:
        cv2.line(step3_img, (x1, y1), (x2, y2), (255, 0, 255), 2) # Magenta
    cv2.imshow('3. Raw Line Segments', cv2.resize(step3_img, (WINDOW_WIDTH, WINDOW_HEIGHT)))

    grid_lines = merge_and_extend_lines(lines, empty_img.shape)
    step4_img = empty_img.copy()
    for x1, y1, x2, y2 in grid_lines['horizontal']:
        cv2.line(step4_img, (x1, y1), (x2, y2), (255, 255, 0), 2) # Cyan
    for x1, y1, x2, y2 in grid_lines['vertical']:
        cv2.line(step4_img, (x1, y1), (x2, y2), (255, 255, 0), 2) # Cyan
    cv2.imshow('4. Merged & Extended Grid', cv2.resize(step4_img, (WINDOW_WIDTH, WINDOW_HEIGHT)))
    
    parking_spots = generate_spots_from_grid(grid_lines, empty_img.shape)
    filtered_spots = filter_invalid_spots(parking_spots)
    
    spots_with_status = determine_occupancy_status(empty_img, filled_img, filtered_spots)
    
    step5_img = display_spots_with_occupancy(filled_img, spots_with_status)
    cv2.imshow('5. Spots with Occupancy Status', cv2.resize(step5_img, (WINDOW_WIDTH, WINDOW_HEIGHT)))

    print(f"Detected {len(spots_with_status)} parking spots.")
    occupied_count = sum(1 for spot in spots_with_status if spot['occupied'])
    empty_count = len(spots_with_status) - occupied_count
    print(f"Occupied spots: {occupied_count}, Empty spots: {empty_count}")
    

    data = {
        "timestamp": str(cv2.getTickCount()),
        "total_spots": len(spots_with_status),
        "occupied_spots": sum(1 for spot in spots_with_status if spot['occupied']),
        "empty_spots": sum(1 for spot in spots_with_status if not spot['occupied']),
        "spots": spots_with_status
    }
    
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    cv2.imwrite(output_image_path, step5_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return spots_with_status

def detect_white_lines(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    kernel = np.ones((2, 1), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    
    # Remove small noise
    kernel_small = np.ones((1, 1), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_small)
    
    return white_mask

def detect_line_segments(mask):
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=20, minLineLength=25, maxLineGap=10)
    
    horizontal_lines, vertical_lines = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 45 or angle > 135:
                horizontal_lines.append(line[0])
            else:
                vertical_lines.append(line[0])
    return {'horizontal': horizontal_lines, 'vertical': vertical_lines}

def merge_and_extend_lines(lines, img_shape):
    img_height, img_width, _ = img_shape
    merged_lines = {'horizontal': [], 'vertical': []}

    if lines['horizontal']:
        y_coords = np.array([(l[1] + l[3]) / 2 for l in lines['horizontal']]).reshape(-1, 1)
        db = DBSCAN(eps=15, min_samples=1).fit(y_coords)
        for label in set(db.labels_):
            cluster_lines = [lines['horizontal'][i] for i, l in enumerate(db.labels_) if l == label]
            avg_y = int(np.mean([(l[1] + l[3]) / 2 for l in cluster_lines]))
            merged_lines['horizontal'].append((0, avg_y, img_width, avg_y))

    if lines['vertical']:
        x_coords = np.array([(l[0] + l[2]) / 2 for l in lines['vertical']]).reshape(-1, 1)
        db = DBSCAN(eps=15, min_samples=1).fit(x_coords)
        for label in set(db.labels_):
            cluster_lines = [lines['vertical'][i] for i, l in enumerate(db.labels_) if l == label]
            avg_x = int(np.mean([(l[0] + l[2]) / 2 for l in cluster_lines]))
            merged_lines['vertical'].append((avg_x, 0, avg_x, img_height))
    
    return merged_lines

def generate_spots_from_grid(grid_lines, img_shape):
    spots = []
    horizontal = sorted(grid_lines['horizontal'], key=lambda l: l[1])
    vertical = sorted(grid_lines['vertical'], key=lambda l: l[0])
    img_height, img_width, _ = img_shape

    h_lines = [l[1] for l in horizontal]
    v_lines = [l[0] for l in vertical]
    h_lines = sorted(list(set([0] + h_lines + [img_height])))
    v_lines = sorted(list(set([0] + v_lines + [img_width])))
    
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            y1, y2 = h_lines[i], h_lines[i+1]
            x1, x2 = v_lines[j], v_lines[j+1]
            spots.append((x1, y1, x2 - x1, y2 - y1))
    return spots

def filter_invalid_spots(spots):
    filtered = []
    for x, y, w, h in spots:
        MIN_WIDTH, MAX_WIDTH = 40, 150
        MIN_HEIGHT, MAX_HEIGHT = 80, 250
        if (MIN_WIDTH < w < MAX_WIDTH) and (MIN_HEIGHT < h < MAX_HEIGHT):
            filtered.append((x, y, w, h))
    return filtered

def determine_occupancy_status(empty_img, filled_img, parking_spots, threshold=3):
    spots_with_status = []
    
    empty_gray = cv2.cvtColor(empty_img, cv2.COLOR_BGR2GRAY)
    filled_gray = cv2.cvtColor(filled_img, cv2.COLOR_BGR2GRAY)
    
    empty_blur = cv2.GaussianBlur(empty_gray, (5, 5), 0)
    filled_blur = cv2.GaussianBlur(filled_gray, (5, 5), 0)
    
    for i, (x, y, w, h) in enumerate(parking_spots):
        empty_roi = empty_blur[y:y+h, x:x+w]
        filled_roi = filled_blur[y:y+h, x:x+w]
        
        diff = cv2.absdiff(empty_roi, filled_roi)

        diff_normalized = diff.astype(np.float32) / 255.0
        change_percentage = np.mean(diff_normalized) * 100 # percentage from 0 to 100
        
        is_occupied = change_percentage > threshold

        spot_info = {
            'id': int(i + 1),
            'rect': [int(x), int(y), int(w), int(h)],
            'occupied': bool(is_occupied),
            'change_percentage': float(change_percentage),
            'confidence': float(min(abs(change_percentage - threshold) * 2, 1.0))
        }

        spots_with_status.append(spot_info)
    
    return spots_with_status

def display_spots_with_occupancy(img, spots_with_status):
    result_img = img.copy()
    
    for spot in spots_with_status:
        x, y, w, h = spot['rect']
        spot_id = spot['id']
        color = (0, 0, 255) if spot['occupied'] else (0, 255, 0)  # Red for occupied, Green for empty
        thickness = 3 if spot['occupied'] else 2
        
        cv2.rectangle(result_img, (x, y), (x+w, y+h), color, thickness)
        cv2.putText(result_img, str(spot_id), (x + 5, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        # if 'change_percentage' in spot:
        #     conf_text = f"{spot['change_percentage']:.2f}"
        #     cv2.putText(result_img, conf_text, (x + 5, y + 15), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    return result_img

if __name__ == "__main__":
    spots_with_status = detect_parking_spots_with_occupancy(
        empty_lot_path="empty_parking_lot.jpg",
        filled_lot_path="not-empty-lot.png",
        output_json_path="parking_spots_with_occupancy.json",
        output_image_path="detected_spots_with_occupancy.jpg"
    )
    
    if spots_with_status:
        occupied = sum(1 for spot in spots_with_status if spot['occupied'])
        total = len(spots_with_status)
        print(f"Total spots: {total}")
        print(f"Occupied: {occupied}")
        print(f"Available: {total - occupied}")