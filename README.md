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
 ### Solution: [reference](https://leetcode.com/problems/max-increase-to-keep-city-skyline/discuss/120681/Easy-and-Concise-Solution-C++JavaPython)
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
   
## 804. Unique Morse Code Words [(original link)](https://leetcode.com/problems/unique-morse-code-words/description/)  
International Morse Code defines a standard encoding where each letter is mapped to a series of dots and dashes, as follows: "a" maps to ".-", "b" maps to "-...", "c" maps to "-.-.", and so on.

For convenience, the full table for the 26 letters of the English alphabet is given below:  
[".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]  
Now, given a list of words, each word can be written as a concatenation of the Morse code of each letter. For example, "cab" can be written as "-.-.-....-", (which is the concatenation "-.-." + "-..." + ".-"). We'll call such a concatenation, the transformation of a word.

Return the number of different transformations among all words we have.

>Example:
Input: words = ["gin", "zen", "gig", "msg"]  
Output: 2  
Explanation:   
The transformation of each word is:  
"gin" -> "--...-."  
"zen" -> "--...-."  
"gig" -> "--...--."  
"msg" -> "--...--."  
There are 2 different transformations, "--...-." and "--...--.".
  
## Analysis
Use ```dict``` in python to construct the lookup table from letter to Morse code  
Find the corresponding Morse code for each words, and put them in ```set```  
Return the length of the ```set``` we got.  
## Solution
```
class Solution:
    def uniqueMorseRepresentations(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        #construct the dict 
        table = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
        alph = 'abcdefghijklmnopqrstuvwxyz'
        alph_table = {}
        #compute codes
        for i in range(len(alph)):
            alph_table[alph[i]] = table[i]
        codes = [''.join([alph_table[letter] for letter in word]) for word in words]
        return len(set(codes))
```
Time complexity: **O(S)**, where S is the sum of length of word in ```words```  
Space complexity: **O(S)**

## 654. Maximum Binary Tree  [(original link)](https://leetcode.com/problems/maximum-binary-tree/description/)
Given an integer array with no duplicates. A maximum tree building on this array is defined as follow:  
  
1. The root is the maximum number in the array.  
2. The left subtree is the maximum tree constructed from left part subarray divided by the maximum number.  
3. The right subtree is the maximum tree constructed from right part subarray divided by the maximum number.  
Construct the maximum tree by the given array and output the root node of this tree.  
>Example: 
Input: [3,2,1,6,0,5]  
Output: return the tree root node representing the following tree:  
  ![Alt text](https://github.com/ChenWentai/LeetCode-problems/blob/master/images/BinaryTree.PNG)  
    
## Analysis
Given a list ```nums```, use the maximum value as the root node, and the left/right sub-array as the left/right sub-tree. Then do the same operation on the left/right sub-array until all the elements in ```nums``` are added to the tree.  
  
To solve the problem, we first need to define a function ```find_rot(nodes)```. Given a list ```nodes```, return the root and left/right sub-trees. In the function a recursion structure is necessary to traverse all the elements in ```nums```.  
## Solution  [reference](https://leetcode.com/problems/maximum-binary-tree/discuss/142430/My-Python-Solution:-how-do-you-think-about-it)
```
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def constructMaximumBinaryTree(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        def find_root(nodes):
            if not nodes:
                return None
            root = TreeNode(max(nodes))
            idx = nodes.index(root.val)
            l = nodes[0:idx]
            r = nodes[idx+1:len(nodes)]     
            root.left =find_root(l)
            root.right = find_root(r)
            return root
        return find_root(nums)
```


