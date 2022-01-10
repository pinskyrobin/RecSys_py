from UserCF import recommend as ucfr
from ItemCF import recommend as icfr
import read_dataset as rd

if __name__ == '__main__':
    user_movie = rd.get_um_map(1)
    rating_map = rd.get_rating_map()
    print(icfr(1, user_movie, rating_map, 10, 20))
