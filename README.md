- Thêm API_KEY
- Thêm video "xuanthuy.mp4" vào thư mục inference_service/data
- Chạy model: python .\workflow\main.py --source_video_path="inference_service/data/xuanthuy.mp4"
- Cài đặt uvicorn: pip install uvicorn[standard] fastapi
- Chạy backend: uvicorn traffic_dashboard.main:app --reload --port 8000
- Chạy dashboard: streamlit run traffic_dashboard/dashboard_app.py

