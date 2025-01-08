
## https://en.wikipedia.org/wiki/Bernoulli_trial
### https://en.wikipedia.org/wiki/Binomial_coefficient
### Binomial Coefficient: (n/k) = n!/(k!*(n-k)!)

import math

def get_binomial_coeff(n, k):
	result = (math.factorial(n))/((math.factorial(k) * (math.factorial(n - k))))
	print("binomial coefficient = {0}".format(result))
	return result

def bernoulli_trial(n, k):
	result = get_binomial_coeff(n, k) * (pow(p,k)) * (pow(q,n-k))
	print("probability = {0}".format(result))
	return result

## Coin Toss: probability of 2 tosses out of 4 being heads
# ~ p = 1/2
# ~ q = 1/2
# ~ get_binomial_coeff(4, 2)

## Dice: probability of rolling 2 of same value out of 3 rolls
# ~ p = 1/6
# ~ q = 5/6
# ~ get_binomial_coeff(3, 2)

## Megamillions: probability of matching 5 balls out of 6 drawn
p = 1/70
# ~ q = 69/70
q = 1 - p
# ~ get_binomial_coeff(6, 5)

# Driver program
if __name__ == "__main__":
	# ~ get_binomial_coeff(4, 2)
	bernoulli_trial(6, 5) 
	# ~ bernoulli_trial(25, 1)
