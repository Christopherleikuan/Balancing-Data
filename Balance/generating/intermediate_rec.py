def intermediate_rec(vector1,vector2):
    d = 0.25
    random_num = random.uniform(d,1.0)
    vector_interim = ((vector1 - vector2) * (vector1 - vector2)) ** 0.5
    print "vector_interim\n",vector_interim,"random",random_num
    vector_interim = random_num * vector_interim
    new_bug = vector1 + vector_interim
    return new_bug
