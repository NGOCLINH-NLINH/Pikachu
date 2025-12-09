import cv2
import numpy as np
import matplotlib.pyplot as plt

scale = 1
padding = 1000

# Load video
cap = cv2.VideoCapture("inference_service/data/xuanthuy.mp4")

# Đọc frame đầu tiên
ret, frame = cap.read()
if not ret:
    print("Không thể đọc frame. Vui lòng kiểm tra đường dẫn file video.")
    exit()


frame_resized = cv2.resize(frame, None, fx=scale, fy=scale)

# Lấy kích thước ảnh gốc sau resize (ảnh sẽ được sử dụng làm hệ quy chiếu)
height_resized, width_resized, _ = frame_resized.shape
print(f"Kích thước Ảnh Gốc (Hệ quy chiếu): Chiều rộng = {width_resized} px, Chiều cao = {height_resized} px")
print("-" * 50)

# Thêm padding (đen) để mở rộng canvas cho phép chấm điểm ngoài vùng ảnh
padded = cv2.copyMakeBorder(
    frame_resized,
    padding, padding, padding, padding,  # top, bottom, left, right
    cv2.BORDER_CONSTANT,
    value=[0, 0, 0]
)

# Kích thước ảnh sau khi thêm padding
height_padded, width_padded, _ = padded.shape
print(f"Kích thước Canvas (Sau Padding): Chiều rộng = {width_padded} px, Chiều cao = {height_padded} px")

# Hiển thị ảnh để chấm điểm
plt.figure(figsize=(8, 12))
plt.imshow(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
plt.title(
    "Click 4 points theo thứ tự TL → TR → BR → BL\n(Gốc tọa độ [0,0] của kết quả là góc trên trái của khung hình màu)")
plt.axis("off")

# Người dùng click 4 điểm trên ảnh PADED
clicked = plt.ginput(4)
plt.close()

# Convert về tọa độ thật (hệ quy chiếu của ảnh gốc)
# Tọa độ gốc (0, 0) của ảnh PADED là góc trên trái của khung đen.
# Ảnh gốc nằm tại vị trí (padding, padding) trên ảnh PADED.
pts_real = []
for (x_padded, y_padded) in clicked:
    # 1. Dịch chuyển tọa độ click về gốc tọa độ của ảnh gốc (trừ padding)
    x_shifted = x_padded - padding
    y_shifted = y_padded - padding

    # 2. Loại bỏ scale (chia cho scale)
    # Vì bạn muốn tọa độ real, ta chia cho scale (nếu scale=1 thì không thay đổi)
    x_real = int(x_shifted / scale)
    y_real = int(y_shifted / scale)

    pts_real.append((x_real, y_real))

print("\n--- KẾT QUẢ TỌA ĐỘ THEO HỆ TRỤC GỐC ---")
print("SOURCE (Tọa độ gốc [0,0] là góc trên trái ảnh gốc, cho phép giá trị âm):")
print(np.array(pts_real))

cap.release()
cv2.destroyAllWindows()