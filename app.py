import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



def calc_similarity(actual, calc):
    value = (abs(actual - calc) / actual) * 100
    return abs(100 - value)



def bar_plot(plot_df,track):
    Track = f"Track {track}"
    temp_df = plot_df[plot_df["Track"] == Track]
    sns.set(rc={'axes.facecolor':'#0E1117', 'figure.facecolor':'#0E1117', 'text.color':'white', 'grid.color':'grey', 'axes.edgecolor':'grey'})
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Category", y="Value", hue="Type", palette=["black", "teal"], data=temp_df)
    plt.xlabel("Category", fontsize=16, color="white")
    plt.ylabel("Value", fontsize=16, color="white")
    plt.title("Similarity", fontsize=17, color="white")
    plt.xticks(rotation=45,fontsize=16, color="white")
    plt.yticks(color = "white")
    st.pyplot(plt)


def plot_popularity(artist,df):
    artist_record = df[df["track_artist"]==artist]
    mean_pop = artist_record.groupby("track_album_release_year")["track_popularity"].mean().reset_index()
    track_per_year = artist_record.groupby('track_album_release_year').size().reset_index(name='count')
    mean_pop["Artist Popularity"] = mean_pop["track_popularity"]*10
    mean_pop["Number of Releases"] = track_per_year["count"]
    st.line_chart(x="track_album_release_year",y=["Artist Popularity","Number of Releases"],x_label="Year",color=["#5c95ff","#d91e36"],data=mean_pop)


def famous_track(record, df):
    df_temp = df[df["track_artist"] == record.iloc[0,1]]
    max = df_temp[df_temp["track_popularity"] == df_temp["track_popularity"].max()]["track_name"]
    st.write("**Most Popular Artist Track (or Tracks)**",)
    max


def about_track(record,df):
    st.markdown("---")
    year = f"{record.iloc[0,19]}"
    duration = f"{(record.iloc[0,18]/60000).round()} min"
    st.markdown("<h2>ABOUT</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 4], gap="small")
    with col1:
        st.markdown(f"**ARTIST**<br>{record.iloc[0,1]}", unsafe_allow_html=True)
        st.markdown(f"**GENRE**<br>{record.iloc[0,5]}", unsafe_allow_html=True)
        st.markdown(f"**SUB-GENRE**<br>{record.iloc[0,6]}", unsafe_allow_html=True)
        st.markdown(f"**YEAR**<br>{year}", unsafe_allow_html=True)
        st.markdown(f"**DURATION**<br>{duration}", unsafe_allow_html=True)
    with col2:
        st.write(f"***Artist Popularity and Number of Tracks over the years***", unsafe_allow_html=True)
        plot_popularity(record.iloc[0,1],df)
    famous_track(record,df)


def overall_sim(score):
    sum = 0
    cnt = 0
    for val in score:
        sum+=score[val]
        cnt+=1
    return sum/cnt


def create_tracks(similar_tracks, score, plot_df):
    st.markdown("---")
    idx = 0
    for index, row in similar_tracks.iterrows():
        with st.container(border=True):
            st.write(f"<h1>{idx+1}.&nbsp;&nbsp;{row['track_name']}</h1>",unsafe_allow_html=True)

            with st.container(border=True):
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.markdown(f"**ARTIST**<br>{similar_tracks.iloc[idx,1]}", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"**GENRE**<br>{similar_tracks.iloc[idx,5]}", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"**SUB-GENRE**<br>{similar_tracks.iloc[idx,6]}", unsafe_allow_html=True)
                with col4:
                    st.markdown(f"**YEAR**<br>{similar_tracks.iloc[idx,19]}", unsafe_allow_html=True)
                with col5:
                    st.markdown(f"**DURATION**<br>{(similar_tracks.iloc[idx,18]/60000).round()}", unsafe_allow_html=True)
            
            col1, col2 = st.columns([1,2])
            with col1:
                with st.container(border=True):
                    st.write("<h4 style='color:teal'>SIMILARITY</h4>",unsafe_allow_html=True)
                    for feature in score[idx]:
                        value = score[idx][feature].round()
                        st.write(f"{feature.capitalize()} &nbsp;&nbsp; - &nbsp;&nbsp; {value}&nbsp;%",unsafe_allow_html=True)
                with st.container(border=True):
                    st.write(f"**MEAN SIMILARITY**&nbsp;&nbsp;&nbsp;&nbsp;***{overall_sim(score[idx]).round(2)}%***",unsafe_allow_html=True)
            with col2:
                with st.container(border=True):
                    bar_plot(plot_df,idx)
        idx = idx+1


def plot_features(array):
    st.bar_chart(array.T, color="#FF4B4B", horizontal=True)


def topArtistSongs(df,song):
    df_features = df[df["track_name"] == song].iloc[0,7:17]
    features_array = df_features.to_numpy().reshape(1, 10)
    features_array = pd.DataFrame(features_array, columns=["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness","liveness","valence"])
    st.write(features_array)
    plot_features(features_array)


def findTopSongs(genres,df):
    for genre in genres:
        df_genre = df[df["playlist_genre"] == genre].sort_values(by="track_popularity",ascending=False).iloc[:3,:]
        idx = 0
        with st.container(border=True):
            st.write(f"<h1 style='color:#5c95ff'>{genre.upper()}</h1>",unsafe_allow_html=True)
            for index ,row in df_genre.iterrows():
                st.write(f"<h2>{idx+1}.&nbsp;&nbsp;{row['track_name']}</h2>",unsafe_allow_html=True)
                popularity = f"{(df_genre.iloc[0,2]*100).round(2)}"
                col1, col2 = st.columns([1,2],vertical_alignment="top")
                with col1:
                    with st.container(border=True):
                        st.write(f"**<h5>POPULARITY**</h5> <h3 style='color:teal'>***{popularity}%***</h3>",unsafe_allow_html=True)
                    with st.container(border=True):
                        st.markdown(f"**ARTIST**<br>{df_genre.iloc[0,1]}", unsafe_allow_html=True)
                        st.markdown(f"**ALBUM**<br>{df_genre.iloc[0,3]}",unsafe_allow_html=True)
                        st.markdown(f"**SUB-GENRE**<br>{df_genre.iloc[0,6]}", unsafe_allow_html=True)
                        st.markdown(f"**YEAR**<br>{df_genre.iloc[0,19]}", unsafe_allow_html=True)
                        # st.markdown(f"**DURATION**<br>{(df_genre.iloc[0,18]/60000).round()} min", unsafe_allow_html=True)
                with col2:
                    with st.container(border=True):
                        st.write(f"<h4>SONG FEATURES</h4>",unsafe_allow_html=True)
                        # st.write("<hr></hr>",unsafe_allow_html=True)
                        topArtistSongs(df_genre,row['track_name'])
                idx+=1
                if(idx!=3):
                    st.write("<hr></hr>",unsafe_allow_html=True)


def plotGenreYears(df):
    genre_popularity = df.groupby(['track_album_release_year', 'playlist_genre'])['track_popularity'].mean().reset_index()
    genre_popular = genre_popularity.groupby('track_album_release_year')['track_popularity'].idxmax()
    most_popular_genre = genre_popularity.loc[genre_popular]

    sns.set(rc={'axes.facecolor':'#0E1117', 'figure.facecolor':'#0E1117', 'text.color':'white', 'grid.color':'grey', 'axes.edgecolor':'grey'})
    
    fig = plt.figure(figsize=(10,5))
    sns.scatterplot(x=most_popular_genre["track_album_release_year"], y=most_popular_genre["playlist_genre"], hue=most_popular_genre["playlist_genre"],
                    palette="Blues",s=110)
    plt.xlabel("release year",color="white")
    plt.xticks(color="white")
    plt.ylabel("genre",color="white")
    plt.yticks(color="white")
    plt.title("Popular genre over the years")
    st.pyplot(fig)



def plotGenre(df,year):
    df_year = df[df["track_album_release_year"] == year]
    df_year_genre = df_year.groupby("playlist_genre")["track_popularity"].mean().reset_index()
    df_year_genre = df_year_genre.sort_values(by = "track_popularity", ascending = False)
    df_year_subgenre = df_year.groupby("playlist_subgenre")["track_popularity"].mean().reset_index().sort_values(by = "track_popularity", ascending = False)
    st.write(f"<h4 style='color:#f7f7de'>Year {year}</h4>",unsafe_allow_html=True)
    st.write(f"Most Popular Genre - <b style='color:salmon'>{df_year_genre.iloc[0,0]}</b>",unsafe_allow_html=True)
    st.bar_chart(data=df_year_genre, x="playlist_genre", y="track_popularity", x_label="Genre", y_label="Average Popularity", color="#5c95ff")
    st.write(f"Most Popular SubGenre - <b style='color:salmon'>{df_year_subgenre.iloc[0,0]}</b>",unsafe_allow_html=True)
    st.bar_chart(data=df_year_subgenre, x="playlist_subgenre", y="track_popularity",horizontal=True, x_label="Sub-Genre",y_label="Average Popularity")



def plotArtist(df,year):
    df_year = df[df["track_album_release_year"] == year]
    df_artist = df_year.groupby(["track_artist","track_name","playlist_genre"])["track_popularity"].mean().reset_index()
    df_artist = df_artist.sort_values(by="track_popularity",ascending=False)
    top_artist = df_artist.iloc[0,0]
    df_artist_info = df[(df["track_album_release_year"] == year) & (df["track_artist"] == top_artist)]
    top_genre = df_artist_info.groupby("playlist_genre")["track_popularity"].mean().reset_index().sort_values(by="track_popularity",ascending=False)
    st.write(f"<h4 style='color:#f7f7de'>Year {year}</h4>",unsafe_allow_html=True)
    with st.container(border=True):
        st.write(f"Top Artist&nbsp;&nbsp;-&nbsp;&nbsp;<b style='color:salmon;font-size:20px'>{top_artist}</b> ✨",unsafe_allow_html=True)
        st.write(f"Track List of Year {year}&nbsp;&nbsp;&nbsp;&nbsp;")
        st.write(df_artist_info[["track_name","track_album_name","playlist_genre","playlist_subgenre"]])
        st.write(f"Genre of artist track (or tracks) with most popularity&nbsp;&nbsp;-&nbsp;&nbsp;<b style='color:salmon;font-size:18px'>{top_genre.iloc[0,0]}</b>",unsafe_allow_html=True)
        top_genre
    with st.container(border=True):
        st.write(f"<h5>TOP ARTISTS OF YEAR {year} AND THE GENRE OF RELEASES</h5>" ,unsafe_allow_html=True)
        col1, col2 = st.columns(2, vertical_alignment="center")
        with col1:
            df_top_artists = df_artist.iloc[:10,0:3]
            df_top_artists
        with col2:
            genre_counts = df_top_artists["playlist_genre"].value_counts()
            fig, ax = plt.subplots()
            custom_colors = ['#0f4c81','#f87575','#5c95ff','#ffcc99','#7e6c6c','#b9e6ff']
            genre_counts.plot.pie(ax=ax, autopct='%1.1f%%', figsize=(6, 6), ylabel='',colors=custom_colors, textprops={'fontsize': 14})
            st.pyplot(fig)
            st.caption("<h5 style='text-align:center'>Genre Distribution<h5>",unsafe_allow_html=True)



def featureComparison(df,year):
    features = ["danceability", "energy", "liveness", "tempo", "loudness", "acousticness", "speechiness","instrumentalness"]
    df_years = df.groupby("track_album_release_year")[features].mean().reset_index()
    st.write("<br></br>",unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    for feature in features:
        sns.lineplot(data=df_years, x="track_album_release_year", y=feature, ax=ax, label=feature)
    ax.set_title(f"Feature Comparison Over Years", fontsize=16)
    ax.set_xlabel("Year", fontsize=14, color="white")
    ax.set_ylabel("Feature Value", fontsize=14, color="white")
    ax.legend(title="Features")
    plt.xticks(rotation=45,color="white")
    plt.yticks(color="white")
    st.pyplot(fig)



def plotFeature(df,feature,genre):
    df_year_genre = df.groupby(["track_album_release_year","playlist_genre"])[feature].mean().reset_index()
    df_genre = df_year_genre[df_year_genre["playlist_genre"] == genre]
    st.write(f"<h4 style='text-align:center'>Feature - {feature}&nbsp;&nbsp;&nbsp;&nbsp; Genre - {genre}</h4>",unsafe_allow_html=True)
    st.line_chart(data=df_genre, x="track_album_release_year", y=feature,x_label="Year")



def predictGenre(X,df,feature):
    # st.write(df[feature].max())
    encoding = {0:"edm", 1:"latin", 2:"pop", 3:"r&b", 4:"rap", 5:"rock"}
    X1 = df[feature]
    y = df["playlist_genre_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(X1,y, test_size = 1/3, random_state=42)
    forestModel = RandomForestClassifier(n_estimators=500,random_state=42)
    forestModel.fit(X_train,y_train)
    sample = np.array([X]).reshape(1, -1)
    val = forestModel.predict(sample)[0]
    genre = encoding[val]
    df_year_genre = df.groupby(["track_album_release_year","playlist_genre"])[feature].mean().reset_index()
    df_genre = df_year_genre[df_year_genre["playlist_genre"] == genre]
    df_mean = df_genre[feature].mean()

    user_data = pd.Series(X, index=feature, name='User')
    mean_data = pd.Series(df_mean, name='Genre Mean')
    plot_data = pd.DataFrame([user_data, mean_data]).T
    plot_data.columns = ['User', 'Genre Mean']
    with st.container(border=True):
        st.write("<h1>RESULT</h1>",unsafe_allow_html=True)
        st.caption("Genre")
        st.write(f"<h2 style='color:#FF4B4B;margin-top:-40px'>{genre}</h2>",unsafe_allow_html=True)
        st.bar_chart(plot_data)
        st.write("But don't base all your assumptions based on the mean value. Beware of OUTLIERS")
        fig, ax = plt.subplots(figsize=(11,5))
        sns.boxplot(df[df["playlist_genre"]==genre][feature], ax=ax, color='skyblue',flierprops=dict(marker='o', markerfacecolor='red', markersize=8, linestyle='none'))
        ax.set_title(f'Box plot for {genre} genre')
        ax.set_xlabel("features")
        ax.tick_params(axis='both', colors='white')
        ax.set_xlabel("features", color='white')
        ax.set_ylabel('Values', color='white')
        st.pyplot()
        st.caption("The red circles above and below the boxes are the outliers. These values have a huge effect on the mean value. In those cases it is better to take the median values into consideration. The median is represented by the grey line in the boxes.")
        st.error("You are now equipped with all the tools to create the next hit of our generation. Tune your instruments and the rest is history 🚀")



def plotCorr(df):
    features = ["track_popularity", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness"]
    genre_option = st.selectbox("Choose a genre for observing feature Correlation",["pop", "rap", "rock", "latin","r&b","edm"])
    df_genre = df[df["playlist_genre"] == genre_option][features]
    corr_matrix = df_genre.corr()
    # st.write(corr_matrix)
    fig, ax = plt.subplots(figsize=(7,7))
    figure = sns.heatmap(corr_matrix, annot=True, fmt=".2f",square=True, cmap="YlGnBu", ax=ax, 
            annot_kws={"size": 10, "weight": "bold", "color":"white"})
    ax.set_title(f"Correlation between {genre_option} genre and Song Features")
    ax.set_xlabel("features",color="white")
    ax.tick_params(axis="both", colors="white")
    ax.set_ylabel("features",color="white")

    colorbar = figure.collections[0].colorbar
    colorbar.ax.tick_params(labelcolor='white')
    st.pyplot(fig)

    st.divider()
    with st.container():
        st.write(f"<h5>CORRELATION ANALYSIS📈</h5>",unsafe_allow_html=True)
        st.write(f"<i><b style='color:salmon'>{genre_option} genre</b> - POSITIVE CORRELATION WITH POPULARITY</i>",unsafe_allow_html=True)
        cnt = 0
        col1, col2 = st.columns([1,1])
        with col1:
            st.write(f"<h6 style='color:salmon'>FEATURES</h6>",unsafe_allow_html=True)
        with col2:
            st.write(f"<h6 style='color:salmon'>CORRELATION</h6>",unsafe_allow_html=True)
        for feature in corr_matrix.iloc[:,0]:
            if((feature>0) & (feature!=1)):
                col1, col2 = st.columns([1,1])
                with col1:
                    with st.container(border=True):
                        st.write(features[cnt])
                with col2:
                    with st.container(border=True):
                        st.write(f"{feature:.2f}")
            cnt+=1





# Main function to run the Streamlit app
def main():
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Explore", ("Song Recommendations :pushpin:", "Trend Analysis :pushpin:"))
    
    st.sidebar.write("<br></br>",unsafe_allow_html=True)
    st.sidebar.write("<br></br>",unsafe_allow_html=True)
    st.sidebar.write("<br></br>",unsafe_allow_html=True)
    st.sidebar.write("<br></br>",unsafe_allow_html=True)
    
    st.sidebar.error("Ready to Explore? ❤️")
    st.sidebar.warning("Made by itsmelps <3")

    # Load data
    df = pd.read_csv("songs.csv")
    features = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness"]

    if options == "Song Recommendations :pushpin:":
        st.title("Hey There! 👋")
        st.write("""
            <div style="color:salmon;font-size:18px">Ready for some new tunes?</div>
            <br>
            """, unsafe_allow_html=True)
        st.write("If you want to explore new songs, you have come to the right place. You can either get recommendations based on a song you like or a genre you have been enjoying recently.")
        st.write("<div></div>",unsafe_allow_html=True)
        options_song = st.radio("What kind of Recommendation would you like today?", ["Song Based🌟", "Genre Based🌟"])
        st.subheader('', divider='rainbow')
        st.write("<br></br>",unsafe_allow_html=True)

        if(options_song == "Song Based🌟"):
            st.title("Song based Recommendations")
            # Select a song
            song = st.selectbox("Select a song", df["track_name"])
            with st.spinner("Finding similar songs..."):
                # Define model
                X = df[features]
                knn = NearestNeighbors(n_neighbors=11)
                knn.fit(X)
                # Find the selected song in the dataset
                record = df[df["track_name"] == song]
                record_features = record[features]

                # Find nearest neighbors
                distance, index = knn.kneighbors(record_features)
                similar_tracks = df.iloc[index[0][1:]]
                # st.write(similar_tracks)

                # Calculate similarity scores
                plot_data = []
                similarity_score = []
                for x in range(len(similar_tracks)):
                    track_record = {}
                    for y in features:
                        if y in features:
                            score = calc_similarity(record_features[y].values[0], similar_tracks[y].values[x])
                            track_record[y] = score

                            plot_data.append({
                                "Track": f"Track {x}",
                                "Category": y,
                                "Value": record_features[y].values[0],
                                "Type": "Actual"
                            })
                            plot_data.append({
                                "Track": f"Track {x}",
                                "Category": y,
                                "Value": score / 100,
                                "Type": "Calculated"
                            })
                    similarity_score.append(track_record)

                plot_df = pd.DataFrame(plot_data)

                # Display similar tracks and plot
                about_track(record,df)
                st.markdown("---")
                st.header("Similar Tracks")
                st.write(f"<h5 style='color:#5c95ff'>Love the song you are listening to?</h5>",unsafe_allow_html=True)
                st.write("Now you can explore songs similar to you **fav songs**. The recommendation algorithm chooses the ten nearest neighbours to your fav based on features defining the song like Energy and Liveness!")
                st.write(f"***Get ready to elevate your senses, dive into your musical paradise now!***",unsafe_allow_html=True)
                create_tracks(similar_tracks,similarity_score,plot_df)

        elif(options_song == "Genre Based🌟"):
            st.title("Genre based Recommendations")
            st.write("""
                <div style="color:salmon;font-size:18px">Hooked with a genre and want more of it ? Get Top Tracks from your fav genres</div>
                <br>
                """, unsafe_allow_html=True)
            
            options = st.multiselect(
                "Select All your favorite Genres",
                ["pop", "rap", "rock", "latin","r&b","edm"])
            
            findTopSongs(options,df)


    elif options == "Trend Analysis :pushpin:":
        st.title("Trend Analysis📊")
        st.write("Wanting to release a new song or explore the trends in music? Well then you are at the right place. This page will give a detailed analysis on the shifts in music trends and help you know what's the big BOOM now.")
        with st.container():
            st.subheader("Content")
            st.write("📍 Genre Analysis")
            st.write("📍 Artist-Genre Analysis")
            st.write("📍 Song Feature Analysis")
            st.write("📍 Feature Popularity Analysis")
            st.write("📍 Genre Prediction")
        st.subheader('', divider='rainbow')
        year = st.selectbox("Select a year", df["track_album_release_year"].unique())
        with st.container(border=True):
            st.write("<h2 style='color:#f7f7de'>Genre popularity over the years</h2>",unsafe_allow_html=True)
            plotGenre(df,year)
            plotGenreYears(df)
            st.write("⬆️ It is evident that r&b and rock was quite popular back in the days and pop and rap are taking over recently",unsafe_allow_html=True)
        with st.container(border=True):
            st.write("<h2 style='color:#f7f7de'>Artist popularity and Genre</h2>",unsafe_allow_html=True)
            st.write("<div style='color:#ede0d4'>This will showcase the most popular artist of any selected year and the genres of songs that they have released that year. Do these results align with the most popular genre of each year? <h5 style='color:teal'>LETS ANALYZE!</h5></div>",unsafe_allow_html=True)
            plotArtist(df,year)
        with st.container(border=True):
            st.write("<h2 style='color:#f7f7de'>Evolution of Song Features</h2>",unsafe_allow_html=True)
            st.write("<div style='color:#ede0d4'>It is now evident that popular artist make songs that fit the current genre trend. But is that all it takes to make a hit track? Let's explore other music features and how they have evolved over the years</h5></div>",unsafe_allow_html=True)
            featureComparison(df,year)
            st.write("You can observe these features one at a time")
            feature_option = st.selectbox("Choose a feature",features)
            genre_option = st.selectbox("Choose a genre",["pop", "rap", "rock", "latin","r&b","edm"])
            plotFeature(df,feature_option,genre_option)
        with st.container(border=True):
            st.write("<h2 style='color:#f7f7de'>Correlation between Genre and Features</h2>",unsafe_allow_html=True)
            features = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness","instrumentalness"]
            genre = ["pop", "rap", "rock", "latin","r&b","edm"]
            plotCorr(df)

        with st.container(border=True):
            st.write("<h2 style='color:#f7f7de'>Custom Building</h2>",unsafe_allow_html=True)
            st.write("<div style='color:#ede0d4'>Finally, we are ready to cook up a hit track for ourselves. We have observed the shift in the popular genre of music over the years and the features that can be tuned to define a genre. Below is a custom classifier that takes your desired values for the features and predicts the genre of music. </h5></div>",unsafe_allow_html=True)
            st.write("<div></div>",unsafe_allow_html=True)
            st.error("Gear Up for the Musical Revolution! 🎧🎹🎸")
            st.write("<div></div>",unsafe_allow_html=True)
            st.write("<div></div>",unsafe_allow_html=True)
            with st.container(border=True):
                features = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness","instrumentalness"]
                X_test = []
                for feature in features:
                    value = st.slider(label=f"**{feature.upper()}**", min_value=0.0, max_value=1.0, value=0.5, step=0.001)
                    X_test.append(value)
                
                st.write("<div></div>",unsafe_allow_html=True)
                st.write("<div></div>",unsafe_allow_html=True)
                st.write("<div></div>",unsafe_allow_html=True)
                press = st.button(label="⭐PREDICT⭐",type="primary")
                st.write("<div></div>",unsafe_allow_html=True)
            if(press):
                with st.spinner("Loading your Results..."):
                    predictGenre(X_test,df,features)


if __name__ == "__main__":
    main()
