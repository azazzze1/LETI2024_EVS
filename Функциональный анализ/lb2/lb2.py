import numpy as np

def gaussFunc(matrix):
    nM = matrix.copy()
    for nrow, row in enumerate(nM):
        divider = row[nrow] 
        row /= divider
        for lower_row in nM[nrow+1:]:
            factor = lower_row[nrow] 
            lower_row -= factor*row 
    return nM

def modMatrix(A):
    nA = A.copy()
    for i in range(len(A)):
        for j in range(len(A[0])):
            if(nA[i][j] < 0):
                nA[i][j] *= -1

    return nA

def maxRawSumFunc(A):
    maxRawSum = 0
    maxRawIdx = 0
    nA = modMatrix(A)
    
    for rawIdx in range(4):
        print("Столбец\строка " + str(rawIdx) + ": ", np.sum(nA[rawIdx]))
        if(np.sum(nA[rawIdx]) > maxRawSum):
            maxRawSum = np.sum(nA[rawIdx])
            maxRawIdx = rawIdx    
    
    return maxRawSum, maxRawIdx

def iterationMethod(x, b, B, idx = 0):
    print("Шаг ", idx, ": x = ", x)
    if(idx != 10):
        iterationMethod(b + np.dot(B, x), b, B, idx + 1)

A = np.array([[-135/22, -873/11, 1269/22, 783/22],[27/2, 156, -147/2, -129/2],[0, 54, 9, -18],[243/22, 1017/11, -1611/22, -855/22]])
print("Матрица A: ")
print(A)

# Задание 1: норма оператора А в пространствах l_1_4 и l_inf_4

print("\nЗАДАНИЕ 1 --------------------------------------------------")

maxColumnSum, maxColumnIdx = maxRawSumFunc(np.transpose(A))

print("\nНорма A в l_1_4: ", maxColumnSum)
print("Индекс столбца, на котором достигается: ", maxColumnIdx)

maxRawSum, maxRawIdx = maxRawSumFunc(A)

print("\nНорма A в l_inf_4: ", maxRawSum, maxRawIdx)
print("Вектор, на котором достигается: ", A[maxRawIdx])

# Задание 2: норма оператора А^(-1) в пространствах l_1_4 и l_inf_4

print("\nЗАДАНИЕ 2 --------------------------------------------------")

print("Ядро оператора А: ")
print(gaussFunc(A)) 

AInv = np.linalg.inv(A)

print("\nМатрица A^(-1): ")
print(AInv)

maxColumnSumInv, maxColumnIdxInv = maxRawSumFunc(np.transpose(AInv))

print("\nНорма A^(-1) в l_1_4: ", maxColumnSumInv)
print("Индекс столбца, на котором достигается: ", maxColumnIdxInv)

maxRawSumInv, maxRawIdxInv = maxRawSumFunc(AInv)

print("\nНорма A^(-1) в l_inf_4: ", maxRawSumInv, maxRawIdxInv)
print("Вектор, на котором достигается: ", AInv[maxRawIdxInv])

# Задание 3: число обусловленности A и А^(-1) в пространствах l_1_4 и l_inf_4

print("\nЗАДАНИЕ 3 --------------------------------------------------")

condA_l14 = maxColumnSum * maxColumnSumInv
condA_inf4 = maxRawSum * maxRawSumInv

print("\nЧисло обусловленности оператора A в пространстве в l_1_4: ", condA_l14)
print("Число обусловленности оператора A в пространстве в l_inf_4: ", condA_inf4)

# Задание 4: создать матрицу G=A*A, показать, что она положительно определена. Найти её собственные числа и векторы.
## Матрица положительно определена, если все её собственные числа - положительны 

print("\nЗАДАНИЕ 4 --------------------------------------------------")

G = np.dot(np.transpose(A), A) 
print("Матрица G: ")
print(G) 
eigenvalues, eigenvectors = np.linalg.eig(G)
print("\nСобственные числа: ", eigenvalues)
print("Собственные вектора: ", eigenvectors)

# Задание 5: Число обусловленности оператора А в l_2_4

print("\nЗАДАНИЕ 5 --------------------------------------------------")

maxEigValG = np.max(eigenvalues)
maxEigVecG = eigenvectors[eigenvalues.tolist().index(maxEigValG)]

normA = np.sqrt(maxEigValG)

print("\nНорма оператора A в l_2_4 и вектор: ", normA, maxEigVecG)

minEigValG = np.min(eigenvalues)
minEigVecG = eigenvectors[eigenvalues.tolist().index(minEigValG)]

normAInv = 1/np.sqrt(minEigValG)

print("Норма оператора A^(-1) в l_2_4 и вектор: ", normAInv, minEigVecG)

print("\nЧисло обусловленности A в l_2_4: ", normAInv * normA)

# Задание 6: реализовать метод итераций 

print("\nЗАДАНИЕ 6 --------------------------------------------------")

x0 = [1,1,1,1]
b = [1/2, 1/3, 1/4, 1/5]
B = np.linalg.inv(G)
A = np.eye(4) - B

print("Матрица B: ")
print(B)

print("\nМатрица A: ") 
print(A)

maxColumnSum, maxColumnIdx = maxRawSumFunc(np.transpose(B))

print("\nНорма B в l_1_4: ", maxColumnSum)
print("Индекс столбца, на котором достигается: ", maxColumnIdx)

maxRawSum, maxRawIdx = maxRawSumFunc(B)

print("\nНорма A в l_inf_4: ", maxRawSum, maxRawIdx)
print("Вектор, на котором достигается: ", B[maxRawIdx])

print("\nТочное решение СЛУ: ", np.linalg.tensorsolve(A, b), "\n")


iterationMethod(x0, b, B)
