import streamlit as st
import pandas as pd
import psycopg2
from opencage.geocoder import OpenCageGeocode
from dotenv import load_dotenv
import os 

opencage_api_key = os.getenv("OPENCAGE_API_KEY")

geolocator = OpenCageGeocode(opencage_api_key)

if "edited_df" not in st.session_state:
    st.session_state.edited_df = pd.DataFrame()

def fetch_data():
    connection_params = {
        'host': 'localhost',
        'port': '5432',
        'database': 'Address',
        'user': 'postgres',
        'password': 'pgadmin'
    }

    connection = psycopg2.connect(**connection_params)
    cursor = connection.cursor()

    # query = "SELECT insured_legal_name, reporting_unit_name, store_number, op_unit, address_line1, city, state, zipcode, county, location FROM userdata;"
    
    query="SELECT * FROM userdata ORDER BY ID ASC;"
    cursor.execute(query)
    data = cursor.fetchall()

    column_names = [desc[0] for desc in cursor.description]

    connection.close()

    return pd.DataFrame(data, columns=column_names)

def update_location(address, city, state, zipcode, country):
    full_address = f"{address}, {city}, {state} {zipcode}, {country}"
    location = geolocator.geocode(full_address)
    if location and 'geometry' in location[0]:
        geometry = location[0]['geometry']
        lat = geometry['lat']
        lng = geometry['lng']
        return lat, lng
    else:
        return 0, 0

def update_data(original_data, updated_data):
    connection_params = {
        'host': 'localhost',
        'port': '5432',
        'database': 'Address',
        'user': 'postgres',
        'password': 'pgadmin'
    }

    connection = psycopg2.connect(**connection_params)
    cursor = connection.cursor()

    for index, row in updated_data.iterrows():
        original_row = original_data.iloc[index]
        
        if not row.equals(original_row):
            latitude, longitude = update_location(row['address_line1'], row['city'], row['state'], row['zipcode'], row['county'])
            if latitude is not None and longitude is not None:
                query = f"UPDATE userdata SET insured_legal_name = '{row['insured_legal_name']}', reporting_unit_name = '{row['reporting_unit_name']}', store_number = '{row['store_number']}', op_unit = '{row['op_unit']}', address_line1 = '{row['address_line1']}', city = '{row['city']}', state = '{row['state']}', zipcode = '{row['zipcode']}', county = '{row['county']}', location = POINT({latitude}, {longitude}) WHERE id = {row['id']};"
                cursor.execute(query)

    connection.commit()
    connection.close()


def main():
   


    st.set_page_config(page_title='Dynamic DataBase', page_icon=':clipboard:', layout='wide', initial_sidebar_state='auto')

    

    st.title('Exposure Management')
    
    
    st.sidebar.image("./genpactlogo.png")
    st.sidebar.header("Description")
    st.sidebar.write("Explore our interactive website where information dynamically adjusts based on your location.To update your address, Locate the 'Address' field, and enter your new address. The system will automatically update the latitude and longitude based on the provided address. Remember to save your changes before leaving the page. If you encounter any issues, feel free to reach out to our support team for assistance.")

 
    st.markdown("""
        <style>
            .ef3psqc11 {
                background-color: #9DCBD6 !important;
                color: #000000 !important;
            }
        </style>
    """, unsafe_allow_html=True)
    data = fetch_data()
    updated_data = st.data_editor(data)

    col1, col2, col3 = st.columns([2, 2, 3]) 
    with col1:
        if st.button('Save',key='custom_button_key'):
            update_data(data, updated_data)
            st.success('Data updated successfully!')
    
if __name__ == '__main__':
    main()
