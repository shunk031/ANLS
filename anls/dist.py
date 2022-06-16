def levenshtein_distance(s1: str, s2: str) -> int:

    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = list(range(len(s1) + 1))
    for i2, c2 in enumerate(s2):
        dists = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                dists.append(distances[i1])
            else:
                dists.append(1 + min((distances[i1], distances[i1 + 1], dists[-1])))
        distances = dists

    return distances[-1]


def normalized_levenshtein_distance(s1: str, s2: str) -> float:
    dist = levenshtein_distance(s1, s2)
    length = max(len(s1.upper()), len(s2.upper()))
    return 0.0 if length == 0 else dist / length
