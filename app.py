from MovieLens import MovieLens
from surprise import SVD, KNNBasic
import flask
import heapq
from collections import defaultdict
from operator import itemgetter

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

def recmovieKNN(algo):

    testSubject = '85'
    k = 10

    # Load our data set and compute the user similarity matrix
    ml = MovieLens()
    data = ml.loadMovieLensLatestSmall()

    trainSet = data.build_full_trainset()
    #model = KNNBasic(sim_options = {'name': 'cosine', 'user_based': True})
    algo.fit(trainSet)
    simsMatrix = algo.compute_similarities()

    # Get top N similar users to our test subject
    # (Alternate approach would be to select users up to some similarity threshold - try it!)
    testUserInnerID = trainSet.to_inner_uid(testSubject)
    similarityRow = simsMatrix[testUserInnerID]

    similarUsers = []
    for innerID, score in enumerate(similarityRow):
        if (innerID != testUserInnerID):
            similarUsers.append( (innerID, score) )

    kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])

    # Get the stuff they rated, and add up ratings for each item, weighted by user similarity
    candidates = defaultdict(float)
    for similarUser in kNeighbors:
        innerID = similarUser[0]
        userSimilarityScore = similarUser[1]
        theirRatings = trainSet.ur[innerID]
        for rating in theirRatings:
            candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore
    
    # Build a dictionary of stuff the user has already seen
    watched = {}
    names ={}
    years ={}

    for itemID, rating in trainSet.ur[testUserInnerID]:
        watched[itemID] = 1
    
    # Get top-rated items from similar users:
    pos = 0
    for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
        if not itemID in watched:
            movieID = trainSet.to_raw_iid(itemID)
            names[pos] = ml.getMovieName(int(movieID))
            years[pos] = ml.getYear(int(movieID))
            pos += 1
            if (pos > 10):
                break

    return names, years


def recmovie(algo):
   
    testSubject = 85

    ml = MovieLens()

    data = ml.loadMovieLensLatestSmall()
  
    userRatings = ml.getUserRatings(testSubject)
    loved = []

    for ratings in userRatings:
        if (float(ratings[1]) > 4.0):
            loved.append(ratings)

    print("\nUser ", testSubject, " loved these movies:")

    for ratings in loved:    
        print(ml.getMovieName(ratings[0]))

    trainSet = data.build_full_trainset()

    algo.fit(trainSet)

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
    movie, year = recmovie(SVD(n_epochs = 20, lr_all = 0.005, n_factors = 10))
    return flask.render_template('recmovie.html', alg = "SVD" ,movie_names= movie, movie_year = year)

@app.route('/KNN - cosine')
def rec1():
    UserKNN = KNNBasic(sim_options = {'name': 'cosine', 'user_based': True})
    movie, year = recmovieKNN(UserKNN)
    return flask.render_template('recmovie.html' ,alg = "KNN - cosine similarity" ,movie_names= movie, movie_year = year)

@app.route('/KNN - pearson')
def rec2():
    KNN = KNNBasic(sim_options = {'name': 'pearson', 'user_based': True})
    movie, year = recmovieKNN(KNN)
    return flask.render_template('recmovie.html' ,alg = "KNN - pearson correlation",movie_names= movie, movie_year = year)


if __name__ == '__main__':
    app.run(debug=True)