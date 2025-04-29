import numpy as np

# Define fuzzy set operations


def fuzzy_union(A, B):
    """Union of two fuzzy sets A and B"""
    return np.maximum(A, B)


def fuzzy_intersection(A, B):
    """Intersection of two fuzzy sets A and B"""
    return np.minimum(A, B)


def fuzzy_complement(A):
    """Complement of a fuzzy set A"""
    return 1 - A


def demonstrate_de_morgan(A, B):
    """Demonstrate De Morgan's Laws"""
    complement_union = fuzzy_complement(fuzzy_union(A, B))
    complement_intersection = fuzzy_complement(fuzzy_intersection(A, B))

    # De Morgan's Law: Complement of Union = Complement of A intersect Complement of B
    law_1 = np.allclose(
        complement_union, fuzzy_intersection(fuzzy_complement(A), fuzzy_complement(B))
    )

    # De Morgan's Law: Complement of Intersection = Complement of A union Complement of B
    law_2 = np.allclose(
        complement_intersection, fuzzy_union(fuzzy_complement(A), fuzzy_complement(B))
    )

    return complement_union, complement_intersection, law_1, law_2


# Fuzzy sets for demonstration
A = np.array([0.1, 0.4, 0.7, 0.8, 1.0])  # Fuzzy set A
B = np.array([0.3, 0.6, 0.2, 0.9, 0.5])  # Fuzzy set B
C = np.array([0.5, 0.2, 0.4, 0.6, 0.9])  # Fuzzy set C

# Demonstrating the fuzzy set operations

print("Fuzzy Set A:", A)
print("Fuzzy Set B:", B)
print("Fuzzy Set C:", C)

# Union
print("\nUnion of A and B:", fuzzy_union(A, B))

# Intersection
print("\nIntersection of A and B:", fuzzy_intersection(A, B))

# Complement of A
print("\nComplement of A:", fuzzy_complement(A))

# De Morgan's Law Demonstration
complement_union, complement_intersection, law_1, law_2 = demonstrate_de_morgan(A, B)
print("\nComplement of Union A U B:", complement_union)
print("Complement of Intersection A ∩ B:", complement_intersection)
print("\nDe Morgan's Law (Complement of Union):", "Valid" if law_1 else "Invalid")
print("De Morgan's Law (Complement of Intersection):", "Valid" if law_2 else "Invalid")


# =============================================
#              FUZZY LOGIC - VIVA NOTES
# =============================================

# ----------- CRISP SET VS FUZZY SET -----------
# Crisp Set:
# - Membership is binary (0 or 1)
# - Elements either belong or do not belong
# - Example: A = {1, 2, 3}

# Fuzzy Set:
# - Membership is a value in [0, 1]
# - Elements can partially belong to a set
# - Example: A = {(1, 0.2), (2, 0.5), (3, 1.0)}

# ----------- FUZZY SET OPERATIONS -------------

# 1. Union (A ∪ B)
#    μ_A∪B(x) = max(μ_A(x), μ_B(x))
#    Meaning: Highest degree of belonging in either set

# 2. Intersection (A ∩ B)
#    μ_A∩B(x) = min(μ_A(x), μ_B(x))
#    Meaning: Commonality between the sets

# 3. Complement (A')
#    μ_A'(x) = 1 - μ_A(x)
#    Meaning: Degree to which x does not belong to A

# ----------- DE MORGAN’S LAWS -----------------

# 1. (A ∪ B)' = A' ∩ B'
# 2. (A ∩ B)' = A' ∪ B'

# Note: In fuzzy logic, these laws hold approximately due to floating-point math

# ------------ VIVA QUESTIONS ------------------

# Q1: What is a fuzzy set?
# A: A set where elements have degrees of membership between 0 and 1.

# Q2: How is a fuzzy set different from a crisp set?
# A: Crisp sets have 0/1 membership; fuzzy sets allow partial membership.

# Q3: What is the union operation in fuzzy sets?
# A: max(μ_A(x), μ_B(x))

# Q4: What is the intersection operation?
# A: min(μ_A(x), μ_B(x))

# Q5: How is the complement calculated?
# A: μ_A'(x) = 1 - μ_A(x)

# Q6: What are De Morgan’s Laws in fuzzy logic?
# A: (A ∪ B)' = A' ∩ B' and (A ∩ B)' = A' ∪ B'

# Q7: Do De Morgan’s Laws always hold exactly?
# A: No, they hold approximately due to decimal rounding.

# Q8: Why is fuzzy logic used in AI?
# A: For reasoning with uncertain/imprecise info in real-world scenarios.

# Q9: Give a real-life example of a fuzzy set.
# A: "Tall people" – someone 5'10" might have 0.6 membership in "tall".

# Q10: Applications of fuzzy logic?
# A: Washing machines, ACs, traffic lights, trading systems, etc.

# =============================================
#              END OF FUZZY LOGIC NOTES
# =============================================
