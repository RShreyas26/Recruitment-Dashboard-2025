import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import re
import plotly.express as px
import plotly.graph_objects as go
import datetime


# Fetching Application table and converting to csv
## API Key:

API_KEY = st.secrets["API_KEY"]
base_url = "https://harvest.greenhouse.io/v1/applications"
per_page = 100  # max per page

url = f"{base_url}?page=1&per_page={per_page}"
all_applications = []

while url:
    response = requests.get(url, auth=HTTPBasicAuth(API_KEY, ""))
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        break

    data = response.json()
    if not data:
        break

    all_applications.extend(data)

    link_header = response.headers.get('Link', '')
    # Parse link header to find 'next' URL
    next_url = None
    if link_header:
        links = link_header.split(',')
        for link in links:
            parts = link.split(';')
            if len(parts) < 2:
                continue
            url_part = parts[0].strip()[1:-1]  # removes <>
            rel_part = parts[1].strip()
            if rel_part == 'rel="next"':
                next_url = url_part
                break

    url = next_url  # will be None if no next page

df_applications = pd.DataFrame(all_applications)

df_application_active = df_applications[df_applications['status'] == 'active']
# Remove rows where current_stage is missing or not a dict with 'name'
df_application_active = df_application_active[
    df_application_active['current_stage'].apply(lambda x: isinstance(x, dict) and 'name' in x)
]

# Then extract the stage name safely
df_application_active['current_stage_name'] = df_application_active['current_stage'].apply(lambda x: x['name'])
stage_mapping = {
    'Presentation | Panel': 'Face to Face',
    'Initial Technical Screen': 'Hiring Manager Screen',
    'Executive Interview': 'Face to Face'
}

df_application_active['current_stage_name'] = df_application_active['current_stage_name'].map(stage_mapping).fillna(df_application_active['current_stage_name'])


## Job Posts
base_url = "https://harvest.greenhouse.io/v1/job_posts"
per_page = 100  # max per page

url = f"{base_url}?page=1&per_page={per_page}"
all_job_posts = []

while url:
    response = requests.get(url, auth=HTTPBasicAuth(API_KEY, ""))
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        break

    data = response.json()
    if not data:
        break

    all_job_posts.extend(data)

    link_header = response.headers.get('Link', '')
    # Parse link header to find 'next' URL
    next_url = None
    if link_header:
        links = link_header.split(',')
        for link in links:
            parts = link.split(';')
            if len(parts) < 2:
                continue
            url_part = parts[0].strip()[1:-1]  # removes <>
            rel_part = parts[1].strip()
            if rel_part == 'rel="next"':
                next_url = url_part
                break

    url = next_url  # will be None if no next page

df_job_posts = pd.DataFrame(all_job_posts)

df_job_posts["location_name"] = df_job_posts["location"].apply(lambda x: x['name'] if pd.notnull(x) and 'name' in x else None)

# Aliases → full country names
country_aliases = {
    "US": "United States",
    "USA": "United States",
    "U.S.": "United States",
    "UAE": "United Arab Emirates",
    "UK": "United Kingdom",
    "LATAM": "Latin America",
}

# Country → Global region
country_to_region = {
    "United States": "North America",
    "Atlanta":"North America",
    "Miami": "North America",
    "Us":"North America",
    "Canada": "North America",
    "Mexico": "North America",
    "Brazil": "LATAM",
    "Latin America": "LATAM",
    "United Kingdom": "EMEA",
    "London":"EMEA",
    "Germany": "EMEA",
    "France": "EMEA",
    "Bulgaria": "EMEA",
    "Sofia":"EMEA",
    "United Arab Emirates": "EMEA",
    "Dubai":"EMEA",
    "Uae":"EMEA",
    "India": "APAC",
    "Singapore": "APAC",
    "China": "APAC",
    "Australia": "APAC",
    "Bangalore":"APAC",
    "ANZ": "APAC"
}

def normalize(part):
    part = part.strip().title()
    return country_aliases.get(part, part)

def clean_and_classify(location):
    if pd.isnull(location):
        return pd.Series([None, "Other"])

    parts = [normalize(p) for p in re.split(r'[,/]', location)]
    parts = list(dict.fromkeys(parts))  # remove dupes

    # Match known country in parts
    for part in reversed(parts):  # search from right → left
        if part in country_to_region:
            return pd.Series([parts[0], country_to_region[part]])

    return pd.Series([parts[0], "Other"])

# Apply it
df_job_posts[['clean_location', 'global_region']] = df_job_posts['location_name'].apply(clean_and_classify)



#Jobs + Offers
API_KEY = st.secrets["API_KEY"]
base_url = "https://harvest.greenhouse.io/v1/jobs"
per_page = 100

url = f"{base_url}?page=1&per_page={per_page}"
jobs = []

while url:
    response = requests.get(url, auth=HTTPBasicAuth(API_KEY, ""))
    
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        break

    data = response.json()
    if not data:
        break

    jobs.extend(data)

    # Parse Link header for next page
    link_header = response.headers.get("Link", "")
    next_url = None
    if link_header:
        links = link_header.split(",")
        for link in links:
            parts = link.split(";")
            if len(parts) < 2:
                continue
            url_part = parts[0].strip()[1:-1]  # Remove < >
            rel_part = parts[1].strip()
            if rel_part == 'rel="next"':
                next_url = url_part
                break

    url = next_url  # If None, loop stops

print(f"✅ Total offers fetched: {len(jobs)}")
df_jobs = pd.DataFrame(jobs)

df_jobs['office_name'] = df_jobs['offices'].apply(
    lambda x: x[0]['name'] if isinstance(x, list) and x else None
)
# Aliases → full country names
country_aliases = {
    "US": "United States",
    "Us": "United States",
    "USA": "United States",
    "U.S.": "United States",
    "UAE": "United Arab Emirates",
    "UK": "United Kingdom",
    "Uk": "United Kingdom",
    "Latam": "Latin America",
    "ANZ":"Australia and New Zealand",
    "Anz":"Australia and New Zealand"
}

# Country → Global region
country_to_region = {
    "United States": "North America",
    "Atlanta":"North America",
    "Miami": "North America",
    "Us":"North America",
    "Canada": "North America",
    "Mexico": "North America",
    "Brazil": "LATAM",
    "Latin America": "LATAM",
    "United Kingdom": "EMEA",
    "UK":"EMEA",
    "London":"EMEA",
    "Germany": "EMEA",
    "France": "EMEA",
    "Bulgaria": "EMEA",
    "Sofia":"EMEA",
    "United Arab Emirates": "EMEA",
    "UAE":"EMEA",
    "Dubai":"EMEA",
    "Uae":"EMEA",
    "India": "APAC",
    "Singapore": "APAC",
    "China": "APAC",
    "Australia": "APAC",
    "Australia and New Zealand": "APAC",
    "Bangalore":"APAC",
}

def normalize(part):
    part = part.strip().title()
    return country_aliases.get(part, part)

def clean_and_classify(location):
    if pd.isnull(location):
        return pd.Series([None, "Other"])

    parts = [normalize(p) for p in re.split(r'[,/]', location)]
    parts = list(dict.fromkeys(parts))  # remove dupes

    # Match known country in parts
    for part in reversed(parts):  # search from right → left
        if part in country_to_region:
            return pd.Series([parts[0], country_to_region[part]])

    return pd.Series([parts[0], "Other"])

# Apply it
df_jobs[['clean_location', 'global_region']] = df_jobs['office_name'].apply(clean_and_classify)


import requests
import pandas as pd
from requests.auth import HTTPBasicAuth

API_KEY = st.secrets["API_KEY"]
base_url = "https://harvest.greenhouse.io/v1/offers"
per_page = 100

url = f"{base_url}?page=1&per_page={per_page}"
offers_data = []

while url:
    response = requests.get(url, auth=HTTPBasicAuth(API_KEY, ""))
    
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        break

    data = response.json()
    if not data:
        break

    offers_data.extend(data)

    # Parse Link header for next page
    link_header = response.headers.get("Link", "")
    next_url = None
    if link_header:
        links = link_header.split(",")
        for link in links:
            parts = link.split(";")
            if len(parts) < 2:
                continue
            url_part = parts[0].strip()[1:-1]  # Remove < >
            rel_part = parts[1].strip()
            if rel_part == 'rel="next"':
                next_url = url_part
                break

    url = next_url  # If None, loop stops

df_offers = pd.DataFrame(offers_data)



API_KEY = st.secrets["API_KEY"]
base_url = "https://harvest.greenhouse.io/v1/offers"
per_page = 100

url = f"{base_url}?page=1&per_page={per_page}"
offers_data = []

while url:
    response = requests.get(url, auth=HTTPBasicAuth(API_KEY, ""))
    
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        break

    data = response.json()
    if not data:
        break

    offers_data.extend(data)

    # Parse Link header for next page
    link_header = response.headers.get("Link", "")
    next_url = None
    if link_header:
        links = link_header.split(",")
        for link in links:
            parts = link.split(";")
            if len(parts) < 2:
                continue
            url_part = parts[0].strip()[1:-1]  # Remove < >
            rel_part = parts[1].strip()
            if rel_part == 'rel="next"':
                next_url = url_part
                break

    url = next_url  # If None, loop stops

df_offers = pd.DataFrame(offers_data)
df_offers = df_offers[df_offers['status'] == 'accepted']

df_offer_jobs = pd.merge(df_offers, df_jobs, how="left", left_on = 'job_id',right_on = 'id' )

df_offer_jobs = df_offer_jobs[['departments','offices','starts_at','openings']]

df_offer_jobs["Department"] = df_offer_jobs["departments"].apply(
    lambda x: x[0]["name"] if isinstance(x, list) and len(x) > 0 and "name" in x[0] else None
)

df_offer_jobs["Region"] = df_offer_jobs["offices"].apply(
    lambda x: x[0]["name"] if isinstance(x, list) and len(x) > 0 and "name" in x[0] else None
)

df_offer_jobs = df_offer_jobs.drop(columns=['offices','departments'])

df_offer_jobs['Region'] = df_offer_jobs['Region'].replace({'Atlanta':'North America','Bulgaria':'EMEA','India':'APAC','UK':'EMEA','UAE':'EMEA','US':'North America'})


# Average time to hire
df_time = df_offer_jobs[['openings']]

# Make sure this is a list of dicts per row
def extract_opening_fields(opening_list):
    if isinstance(opening_list, list) and len(opening_list) > 0:
        opening = opening_list[0]  # assume first opening is most relevant
        return pd.Series({
            "opened_at": opening.get("opened_at"),
            "closed_at": opening.get("closed_at"),
            "close_reason_name": opening.get("close_reason", {}).get("name")
        })
    else:
        return pd.Series({
            "opened_at": None,
            "closed_at": None,
            "close_reason_name": None
        })

# Apply to your DataFrame
df_openings_extracted = df_time["openings"].apply(extract_opening_fields)

# Join back to main DataFrame (optional)
df_time = pd.concat([df_time, df_openings_extracted], axis=1)

# Convert to datetime
df_time["opened_at"] = pd.to_datetime(df_time["opened_at"])
df_time["closed_at"] = pd.to_datetime(df_time["closed_at"])

df_time = df_time.drop(columns = ['openings'])
df_time['Days to hire'] = df_time["closed_at"] - df_time["opened_at"]
df_time = df_time[df_time['close_reason_name'] == 'Hire - New Headcount']

#Funnel Chart
# Remove rows where current_stage is missing or not a dict with 'name'
df_applications = df_applications[
    df_applications['current_stage'].apply(lambda x: isinstance(x, dict) and 'name' in x)
]

# Then extract the stage name safely
df_applications['current_stage_name'] = df_applications['current_stage'].apply(lambda x: x['name'])

# Then extract the stage name safely
stage_mapping = {
    'Presentation | Panel': 'Face to Face',
    'Initial Technical Screen': 'Hiring Manager Screen',
    'Executive Interview': 'Face to Face',
    'Portfolio Review': 'Face to Face', 
    'Team Interview':'Face to Face',
    'Phone Interview':'Recruiter Screen' 
}









# --- Setup ---
st.set_page_config(layout="centered")  # Optional: full width or centered
st.title("Recruitment Dashboard")


st.header("📊 Application Stages Overview ")

stage_counts = df_application_active['current_stage_name'].value_counts()
labels = stage_counts.index.tolist()
sizes = stage_counts.values.tolist()

# --- Custom Colors (edit if needed) ---
colors = plt.cm.Set3.colors[:len(labels)]  # Pick a color set

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops={'width': 0.4, 'edgecolor': 'white'},
    pctdistance=0.85  # Puts % just outside the ring
)





# --- Style tweaks ---
for text in texts:
    text.set_fontsize(12)
    text.set_fontweight('bold')

for autotext in autotexts:
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

# Make it a circle and remove axis
ax.axis('equal')
plt.tight_layout()
total_candidates = sum(sizes)
ax.text(
    0, 0, f"{total_candidates}\nCandidates",
    ha='center', va='center',
    fontsize=16, fontweight='bold'
)

st.pyplot(fig)
# --- Section Header ---
st.header("🌍 Open Jobs by Region")

# Filter for open jobs (optional — only if needed)
# df_open_jobs = df_job_posts[df_job_posts["status"] == "open"]
# If all are open, skip the filter
df_jobs = df_jobs[df_jobs['status'] == 'open']
total_openings = df_jobs.shape[0]
st.metric(label="📌 Total Open Job Postings", value=total_openings)
df_open_jobs = df_jobs.copy()

# --- Count jobs per region ---
region_counts = df_open_jobs["global_region"].value_counts()
region_labels = region_counts.index.tolist()
region_values = region_counts.values.tolist()

# --- Plot Region-wise Job Count ---
fig2, ax2 = plt.subplots(figsize=(8, 6))
bars = ax2.bar(region_labels, region_values, color=plt.cm.Pastel1.colors[:len(region_labels)])

# Style
ax2.set_ylabel("Number of Open Jobs")
ax2.set_title("Open Jobs by Global Region", fontsize=14, fontweight='bold')
ax2.spines[['top', 'right']].set_visible(False)

# Annotate bar values
for bar in bars:
    height = bar.get_height()
    ax2.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5), textcoords="offset points",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

# --- Show in Streamlit ---
st.pyplot(fig2)


# -------Hires-------
regions = sorted(df_offer_jobs["Region"].dropna().unique())
selected_regions = st.sidebar.multiselect("Select Region(s)", regions, default=regions)

# ✅ Define min/max first
df_offer_jobs["starts_at"] = pd.to_datetime(df_offer_jobs["starts_at"])  
min_date = df_offer_jobs["starts_at"].min().date()
max_date = df_offer_jobs["starts_at"].max().date()

# ✅ Then use them in date_input
start_date, end_date = st.sidebar.date_input("Select date range", [min_date, max_date])
# Step 1: Convert to datetime (if not already)

df_applications['last_activity_at'] = pd.to_datetime(df_applications['last_activity_at']).dt.tz_localize(None)
df_applications = df_applications[pd.notnull(df_applications['last_activity_at'])]

# Step 2: Convert sidebar dates to datetime (if needed)
start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
end_datetime = datetime.datetime.combine(end_date, datetime.time.max)


# --- Stage order ---
stages = ['Application Review', 'Recruiter Screen', 'Hiring Manager Screen', 'Face to Face', 'Offer']

# Filter again, just in case it’s not already filtered above
filtered_applications = df_applications[
    (df_applications['last_activity_at'] >= start_datetime) &
    (df_applications['last_activity_at'] <= end_datetime)
].copy()

# Add stage columns (if not already added above)
filtered_applications.loc[:, 'Application Review'] = filtered_applications['current_stage_name'].isin(
    ['Application Review','Face to Face', 'Hiring Manager Screen', 'Recruiter Screen', 'Offer']
).astype(int)

filtered_applications.loc[:, 'Recruiter Screen'] = filtered_applications['current_stage_name'].isin(
    ['Face to Face', 'Hiring Manager Screen', 'Recruiter Screen', 'Offer']
).astype(int)

filtered_applications.loc[:, 'Hiring Manager Screen'] = filtered_applications['current_stage_name'].isin(
    ['Face to Face', 'Hiring Manager Screen', 'Offer']
).astype(int)

filtered_applications.loc[:, 'Face to Face'] = filtered_applications['current_stage_name'].isin(
    ['Face to Face', 'Offer']
).astype(int)

filtered_applications.loc[:, 'Offer'] = filtered_applications['current_stage_name'].isin(
    ['Offer']
).astype(int)




# Now safe to compute funnel_counts
funnel_counts = filtered_applications[stages].sum()


# --- Filter the DataFrame ---
filtered_df = df_offer_jobs[
    (df_offer_jobs["Region"].isin(selected_regions)) &
    (df_offer_jobs["starts_at"] >= pd.to_datetime(start_date)) &
    (df_offer_jobs["starts_at"] <= pd.to_datetime(end_date))
]

# --- Group and Count ---
dept_counts = filtered_df.groupby("Department").size().reset_index(name="Hires")
dept_counts = dept_counts.sort_values(by="Hires", ascending=False)

# --- Display Output ---
st.subheader("💼 Hires by Department")
st.markdown(f"**Regions:** {', '.join(selected_regions)}")
st.markdown(f"**Date Range:** {start_date} → {end_date}")

# --- Bar Chart ---
st.bar_chart(dept_counts.set_index("Department"))

# --- Optional: Show Table ---
st.dataframe(dept_counts)

# --- Filter df_time by selected regions and date range ---
# Add an index to preserve offer alignment before processing df_time
df_time["closed_at"] = df_time["closed_at"].dt.tz_localize(None)
df_time["opened_at"] = df_time["opened_at"].dt.tz_localize(None)


df_offer_jobs_reset = df_offer_jobs.reset_index(drop=True)
df_offer_jobs_reset["offer_index"] = df_offer_jobs_reset.index

# Also copy index into df_time for safe merge
df_time["offer_index"] = df_offer_jobs_reset.loc[df_time.index, "offer_index"]

# Merge Region from df_offer_jobs_reset
df_time = df_time.merge(df_offer_jobs_reset[["offer_index", "Region"]], on="offer_index", how="left")

# Now filter by selected region and date range
# Now safe to filter
filtered_time_df = df_time[
    (df_time["Region"].isin(selected_regions)) &
    (df_time["closed_at"] >= pd.to_datetime(start_date)) &
    (df_time["closed_at"] <= pd.to_datetime(end_date))
]
# --- Calculate average ---
avg_days = filtered_time_df["Days to hire"].mean()

if pd.notnull(avg_days):
    avg_days_only = round(avg_days.total_seconds() / 86400, 1)
    avg_duration_str = f"{avg_days_only} days"
else:
    avg_duration_str = "N/A"

# --- Display metric ---
st.metric("⏱️ Avg Time to Hire", avg_duration_str)


# --- Stage order ---
stages = ['Application Review', 'Recruiter Screen', 'Hiring Manager Screen', 'Face to Face', 'Offer']

funnel_counts = filtered_applications[stages].sum()

# --- Calculate stage-to-stage pass-through percentages ---
funnel_percentages = funnel_counts.shift(-1) / funnel_counts * 100
funnel_percentages = funnel_percentages.round(1)

# Shift percent labels one stage down so they align with the *destination* stage
funnel_percentages = funnel_percentages.shift(1)

# Replace NaNs with '—'
funnel_percentages = funnel_percentages.fillna('—')
funnel_percentages = funnel_percentages.astype(str) + '%'
# --- Build DataFrame ---
funnel_df = pd.DataFrame({
    'Stage': stages,
    'Count': funnel_counts.values,
    'Pass-Through %': funnel_percentages.values,
    'Label': funnel_percentages.values
})

# --- Plot ---
fig_funnel = go.Figure(go.Funnel(
    y=funnel_df['Stage'],
    x=funnel_df['Count'],
    text=funnel_df['Label'],
    textposition="inside",
    hoverinfo="text",  # 
    marker_color='royalblue',
    opacity=0.85,
    textfont=dict(color='white', size=14)
))

fig_funnel.update_layout(
    title="Hiring Funnel",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showticklabels=False),
    yaxis=dict(title='Stage'),
    margin=dict(l=50, r=50, t=50, b=50)
)




# Show in Streamlit
st.header("🔽 Hiring Funnel")
st.plotly_chart(fig_funnel, use_container_width=True, key="hiring_funnel_chart")
