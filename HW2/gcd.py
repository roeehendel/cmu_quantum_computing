x = 106113609170668254652391269192197757215334846951209743863306173107325600
y = 894128743023837450195367301428030915619905056250057436341104142570200


def euclid(a, b):
    if b == 0:
        return a
    return euclid(b, a % b)


t = euclid(x, y)

print(t, ((x // t) * t) == x, ((y // t) * t) == y)
