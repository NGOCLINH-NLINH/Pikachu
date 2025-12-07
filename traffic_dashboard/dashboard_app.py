from pathlib import Path

import streamlit as st
import requests
import json
import pandas as pd

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(layout="wide", page_title="Hệ thống Giám sát Giao thông AI")


@st.cache_data(ttl=60)
def fetch_violations():
    try:
        response = requests.get(f"{API_BASE_URL}/violations")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Không thể kết nối Backend API: {e}")
        return []


def fetch_violation_detail(violation_id):
    response = requests.get(f"{API_BASE_URL}/violations/{violation_id}")
    response.raise_for_status()
    return response.json()


def get_ai_explanation(plate, speed, limit):
    response = requests.post(f"{API_BASE_URL}/explain",
                             params={"plate_number": plate, "speed": speed, "speed_limit": limit})
    response.raise_for_status()
    return response.json().get("explanation", "Agent không thể tạo giải thích.")


def render_violation_ticket(detail):
    st.header(f"Phiếu phạt: {detail['plate_number']}")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"### Vượt quá Tốc độ: **{detail['exceed_speed']:.2f} km/h**")
        st.metric(label="Tốc độ Vi phạm",
                  value=f"{detail['speed']:.2f} km/h",
                  delta=f"Giới hạn: {detail['speed_limit']} km/h")
        st.info(f"**Địa điểm:** {detail['location']} - **Thời gian:** {detail['timestamp']}")

        # Giả lập hiển thị ảnh bằng chứng
        proof_path = detail.get("proof_image_path")
        if proof_path and Path(proof_path).exists():
            st.image(proof_path, caption="Ảnh bằng chứng vi phạm", use_column_width=True)
        else:
            st.warning("🖼 Không tìm thấy ảnh bằng chứng.")

    with col2:
        st.subheader("Thông tin Chủ xe")
        info = detail['vehicle_info']
        st.markdown(f"**Tên:** {info.get('owner', 'Chưa đăng ký')}")
        st.markdown(f"**SĐT:** {info.get('phone', 'N/A')}")
        st.markdown(f"**Địa chỉ:** {info.get('address', 'N/A')}")
        st.markdown(f"**Loại xe:** {info.get('vehicle_type', 'N/A')}")
        st.markdown("---")

        st.subheader("Giải thích Hành vi Vi phạm (AI)")
        if st.button("Yêu cầu Agent Giải thích luật"):
            with st.spinner('Agent đang tra cứu luật và giải thích...'):
                explanation = get_ai_explanation(
                    plate=detail['plate_number'],
                    speed=detail['speed'],
                    limit=detail['speed_limit']
                )
                st.session_state['explanation'] = explanation

        if 'explanation' in st.session_state:
            st.code(st.session_state['explanation'])


def main():
    st.title("Dashboard Giám sát Vi phạm Tốc độ")

    violations_data = fetch_violations()

    if not violations_data:
        st.warning("Chưa có dữ liệu vi phạm nào được ghi nhận.")
        return

    if 'selected_violation_id' not in st.session_state:
        st.session_state['selected_violation_id'] = None
    if 'explanation' not in st.session_state:
        st.session_state['explanation'] = None

    df = pd.DataFrame(violations_data)
    df_display = df[['id', 'plate_number', 'speed', 'speed_limit', 'exceed_speed', 'location', 'timestamp']].copy()
    df_display.columns = ['ID', 'Biển số', 'Tốc độ', 'Giới hạn', 'Vượt quá', 'Địa điểm', 'Thời gian']

    st.subheader("Danh sách các Vi phạm")

    selected_row = st.dataframe(
        df_display,
        selection_mode="single-row",
        hide_index=True,
        on_select="rerun"
    )

    if selected_row.selection and selected_row.selection["rows"]:
        violation_index = selected_row.selection["rows"][0]
        violation_id = df_display.loc[violation_index, 'ID']

        if st.session_state['selected_violation_id'] != violation_id:
            st.session_state['selected_violation_id'] = violation_id
            st.session_state['explanation'] = None  # Reset giải thích khi chọn vi phạm mới
            st.rerun()

    st.markdown("---")

    if st.session_state['selected_violation_id']:
        try:
            violation_detail = fetch_violation_detail(st.session_state['selected_violation_id'])
            render_violation_ticket(violation_detail)
        except requests.exceptions.RequestException:
            st.error("Không thể lấy chi tiết vi phạm. Kiểm tra lại Backend API.")


if __name__ == "__main__":
    main()