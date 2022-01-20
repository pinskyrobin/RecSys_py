# evaluate the percentage of user-item pairs that appear in rec list
def recall(train_recs, test):
    hit = 0
    for i in range(len(test)):
        hit += len(set(test[i]) & set(train_recs[i])) / len(train_recs[i])
    return hit / len(test)


# evaluate the percentage of rec items that appear in user-item list
def precision(train_recs, test):
    hit = 0
    for i in range(len(train_recs)):
        hit += len(set(train_recs[i]) & set(test[i])) / len(test[i])
    return hit / len(train_recs)


# evaluate the amount of rec items for all users
def coverage(train_recs, test):
    rec_part = set()
    all = set()

    for items in test:
        for item in items:
            all.add(item)

    for recs in train_recs:
        for rec in recs:
            rec_part.add(rec)

    return len(rec_part) / (len(all) * 1.0)
