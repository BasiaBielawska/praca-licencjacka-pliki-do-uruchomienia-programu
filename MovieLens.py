import os
import csv
import sys
import re

from surprise import Dataset
from surprise import Reader

class MovieLens:

    movieID_to_name = {}
    name_to_movieID = {}
    movieID_to_year = {}
    ratingsPath = 'dane/ratings.csv'
    moviesPath = 'dane/movies.csv'
    
    def loadMovieLensLatestSmall(self):

        os.chdir(os.path.dirname(sys.argv[0]))

        ratingsDataset = 0
        self.movieID_to_name = {}
        self.name_to_movieID = {}
        self.movieID_to_year = {}

        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

        ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader)
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        
        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
                movieReader = csv.reader(csvfile)
                next(movieReader)  #Skip header line
                for row in movieReader:
                    movieID = int(row[0])
                    title = row[1]
                    m = p.search(title)
                    movieName = p.sub(' ' ,title)
                    year = m.group(1)
                    
                    self.movieID_to_name[movieID] = movieName
                    self.name_to_movieID[movieName] = movieID
                    self.movieID_to_year[movieID] = year
                    
                    

        return ratingsDataset
     
    def getUserRatings(self, user):
        userRatings = []
        hitUser = False
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                userID = int(row[0])
                if (user == userID):
                    movieID = int(row[1])
                    rating = float(row[2])
                    userRatings.append((movieID, rating))
                    hitUser = True
                if (hitUser and (user != userID)):
                    break

        return userRatings 
    

    def getMovieName(self, movieID):
        if movieID in self.movieID_to_name:
            return self.movieID_to_name[movieID]
        else:
            return ""
        
    def getMovieID(self, movieName):
        if movieName in self.name_to_movieID:
            return self.name_to_movieID[movieName]
        else:
            return 0

    def getYear(self, movieID):
        if movieID in self.movieID_to_year:
            return self.movieID_to_year[movieID]
        else:
            return 0         