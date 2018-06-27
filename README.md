# LeetCode-problems
Solutions and analysis of LeetCode problems  
## 771 Jewels and Stones [(original link)](https://leetcode.com/problems/jewels-and-stones/description/)
You're given strings J representing the types of stones that are jewels, and S representing the stones you have.  Each character in S is a type of stone you have.  You want to know how many of the stones you have are also jewels.

The letters in J are guaranteed distinct, and all characters in J and S are letters. Letters are case sensitive, so "a" is considered a different type of stone from "A".

>Example 1:  
>Input: J = "aA", S = "aAAbbbb"  
>Output: 3

>Example 2:  
>Input: J = "z", S = "ZZ"  
>Output: 0
###  Solution 1: Brute force：[reference](https://leetcode.com/problems/jewels-and-stones/discuss/140933/Python-O(M+N)-Hash-)
```
 def numJewelsInStones(self, J, S):   
     result = 0     
        for j in J:     
            for s in S:     
                if j == s:     
                    result += 1     
     return result  
```
A more pythonic way:
```
 def numJewelsInStones(self, J, S):       
     return sum(s in J for s in S)
```
Time complexity: **O(|J|\*|S|)**  
Space complexity: **O(1)**  
  
 ### Solution 2: Use **set**
 ```
 def numJewelsInStones(self, J, S):   
     f = set(J)     
     return sum([s in f for s in S])    
```
Time complexity: **O(|J|\*|S|)**  
The operation ```a in b``` has different time complexity in **list** and **set**, see here：https://wiki.python.org/moin/TimeComplexity  
```a in b``` in list: O(n)  
```a in b``` in set: O(1)  
  
   
 ## 807. Max Increase to Keep City Skyline [(original link)](https://leetcode.com/problems/max-increase-to-keep-city-skyline/description/)  
 In a 2 dimensional array grid, each value grid[i][j] represents the height of a building located there. We are allowed to increase the height of any number of buildings, by any amount (the amounts can be different for different buildings). Height 0 is considered to be a building as well. 

At the end, the "skyline" when viewed from all four directions of the grid, i.e. top, bottom, left, and right, must be the same as the skyline of the original grid. A city's skyline is the outer contour of the rectangles formed by all the buildings when viewed from a distance. See the following example.

What is the maximum total sum that the height of the buildings can be increased?

>Example:
Input: grid = [[3,0,8,4],[2,4,5,7],[9,2,6,3],[0,3,1,0]]  
Output: 35  
Explanation:   
The grid is:  
[ [3, 0, 8, 4],   
  [2, 4, 5, 7],  
  [9, 2, 6, 3],  
  [0, 3, 1, 0] ]  
The skyline viewed from top or bottom is: [9, 4, 8, 7]  
The skyline viewed from left or right is: [8, 7, 9, 3]  
The grid after increasing the height of buildings without affecting skylines is:  
gridNew = [ [8, 4, 8, 7],  
            [7, 4, 7, 7],  
            [9, 4, 8, 7],  
            [3, 3, 3, 3] ]  

    
 ### Analysis  
 The skyline is the maximum value in each row/column of ```grid```.  
 The element **(i,j)** in ```newGrid``` should be ```min{row[j], col[i]}```, where ```row``` and ```col``` are the skylines of ```grid```.  
 So the result is the sum of difference between each element in ```grid``` and ```newGrid```.  
 In the first traverse, compute ```row``` and ```col```.  
 In the second traverse, set the value of ```newGrid``` based on ```row``` and ```col```.  
 Finally calculate the difference.  
 ### Solution
 ```
 class Solution:
    def maxIncreaseKeepingSkyline(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        rows, cols = list(map(max, grid)), list(map(max, zip(*grid)))
        return sum(min(i, j) for i in rows for j in cols) - sum(map(sum, grid))
 ```  
 Time complexity: **O(n\*n)**
