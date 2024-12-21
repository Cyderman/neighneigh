import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import zipfile
import os

# Load preprocessed data
@st.cache_data
def load_processed_data():
    # Unzip and load the data
    if not os.path.exists("processed_data.pkl"):
        with zipfile.ZipFile("processed_data.zip", "r") as zip_ref:
            zip_ref.extractall()
    data = pd.read_pickle("processed_data.pkl")
    combined_embeddings = pd.read_pickle("combined_embeddings.pkl").values
    known_embeddings = pd.read_pickle("known_embeddings.pkl").values
    known_horse_names = pd.read_pickle("known_horse_names.pkl").squeeze().tolist()
    pivot = pd.read_pickle("pivot.pkl")
    return data, combined_embeddings, known_embeddings, known_horse_names, pivot

# Find neighbors function
def find_neighbors(active_horse_name, data, pivot, combined_embeddings, known_embeddings):
    if active_horse_name not in pivot.index:
        st.error(f"Horse '{active_horse_name}' not found in the dataset.")
        return

    # Get the archetype of the active horse
    active_archetype = data.loc[data['horse_name'] == active_horse_name, 'archetype'].iloc[0]

    # Filter to same-archetype retired horses
    same_archetype_horses = data[
        (data['archetype'] == active_archetype) &
        (data['retired'] == True)
    ]
    same_archetype_horse_names = same_archetype_horses['horse_name']
    same_archetype_indices = pivot.index[pivot.index.isin(same_archetype_horse_names)]
    same_archetype_embeddings = combined_embeddings[pivot.index.get_indexer(same_archetype_indices)]

    # Locate the embedding for the active horse
    active_index_in_pivot = pivot.index.get_loc(active_horse_name)
    active_vector = combined_embeddings[active_index_in_pivot].reshape(1, -1)

    # Train a nearest neighbors model for same-archetype embeddings
    archetype_nn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
    archetype_nn_model.fit(same_archetype_embeddings)

    # Find nearest neighbors
    distances, relative_indices = archetype_nn_model.kneighbors(active_vector, n_neighbors=5)

    st.write(f"### Nearest neighbors for horse: {active_horse_name}")

    # Use the mapping to get the original indices
    for rank, (relative_idx, dist) in enumerate(zip(relative_indices[0], distances[0]), start=1):
        neighbor_name = same_archetype_indices[relative_idx]

        # Exclude the active horse from results
        if neighbor_name == active_horse_name:
            continue

        neighbor_data = data[data['horse_name'] == neighbor_name].iloc[0]

        # Retrieve details for the neighbor
        horse_name = neighbor_data['horse_name']
        url = f"https://photofinish.live/horses/{neighbor_data['horse_id']}"
        start = neighbor_data.get('start', 'N/A')
        speed = neighbor_data.get('speed', 'N/A')
        stamina = neighbor_data.get('stamina', 'N/A')
        finish = neighbor_data.get('finish', 'N/A')
        heart = neighbor_data.get('heart', 'N/A')
        temper = neighbor_data.get('temper', 'N/A')

        # Correct preference stars back to the 0-3 range
        surface_star = neighbor_data.get('surface_weight_norm', 0) * 3
        direction_star = neighbor_data.get('direction_weight_norm', 0) * 3
        condition_star = neighbor_data.get('condition_weight_norm', 0) * 3

        st.markdown(f"""
        #### {rank}. Horse: {horse_name}
        - **Start**: {start}, **Speed**: {speed}, **Stamina**: {stamina}, **Finish**: {finish}
        - **Heart**: {heart}, **Temper**: {temper}
        - **Surface Star**: {surface_star:.2f}, **Direction Star**: {direction_star:.2f}, **Condition Star**: {condition_star:.2f}
        - **Distance (similarity)**: {dist:.3f}
        - [View Horse Profile]({url})
        """)

# Streamlit app
def main():
    st.title("Next Nearest Horse Neigh-bour")

    # Load data
    data, combined_embeddings, known_embeddings, known_horse_names, pivot = load_processed_data()

    # Check if reset button was clicked
    if st.button("Reset"):
        st.session_state.active_horse_name = ""  # Clear the input
        st.experimental_set_query_params()  # Clear URL parameters

    # Text input for horse name
    if "active_horse_name" not in st.session_state:
        st.session_state.active_horse_name = ""

    active_horse_name = st.text_input("Enter your horse name:", st.session_state.active_horse_name)

    # Store the active horse name in the session state
    st.session_state.active_horse_name = active_horse_name

    # Run nearest neighbor search if a name is entered
    if active_horse_name:
        find_neighbors(active_horse_name, data, pivot, combined_embeddings, known_embeddings)

if __name__ == "__main__":
    main()
