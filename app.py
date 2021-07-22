from MovieLens import MovieLens
from surprise import SVD, KNNBasic, CoClustering 
import flask

app = flask.Flask(__name__, template_folder='templates')

def BuildAntiTestSetForUser(testSubject, trainset):
    fill = trainset.global_mean

    anti_testset = []
    
    u = trainset.to_inner_uid(str(testSubject))
    
    user_items = set([j for (j, _) in trainset.ur[u]])
    anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                             i in trainset.all_items() if
                             i not in user_items]
    return anti_testset

def recmovie(algo):
   
    testSubject = 60

    ml = MovieLens()

    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
  
    userRatings = ml.getUserRatings(testSubject)
    loved = []

    for ratings in userRatings:
        if (float(ratings[1]) > 4.0):
            loved.append(ratings)

    print("\nUser ", testSubject, " loved these movies:")

    for ratings in loved:    
        print(ml.getMovieName(ratings[0]))

    print("\nBuilding recommendation model...")
    trainSet = data.build_full_trainset()

    algo.fit(trainSet)

    print("\nComputing recommendations...")
    testSet = BuildAntiTestSetForUser(testSubject, trainSet)
    predictions = algo.test(testSet)

    recommendations = []
    names ={}
    years ={}

    for userID, movieID, actualRating, estimatedRating, _ in predictions:
        intMovieID = int(movieID)
        recommendations.append((intMovieID, estimatedRating))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    i = 0 
    while i < 10:
        for ratings in recommendations[:10]:
            names[i] = ml.getMovieName(ratings[0])
            years[i] = ml.getYear(ratings[0])
            i = i + 1    
        
    return names, years

# Set up the main route
@app.route('/',)
def main():
    return flask.render_template('index.html')

@app.route('/SVD')
def rec():
    movie, year = recmovie(SVD())
    return flask.render_template('recmovie.html', alg = "SVD" ,movie_names= movie, movie_year = year)

@app.route('/KNN')
def rec1():
    ItemKNN = KNNBasic(sim_options = {'name': 'cosine', 'user_based': False})
    movie, year = recmovie(ItemKNN)
    return flask.render_template('recmovie.html' ,alg = "KNN" ,movie_names= movie, movie_year = year)

@app.route('/Co-Clustering')
def rec2():
    movie, year = recmovie(CoClustering())
    return flask.render_template('recmovie.html' ,alg = "Co-Clustering",movie_names= movie, movie_year = year)


if __name__ == '__main__':
    app.run(debug=True)




