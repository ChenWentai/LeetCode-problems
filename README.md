# LeetCode problems
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
  
To solve the problem, we first need to define a function ```find_rot(nodes)```. Given a list ```nodes```, return the root and left/right sub-trees. In the function a **recursion** structure is necessary to traverse all the elements in ```nums```.  
## Solution  [(reference)](https://leetcode.com/problems/maximum-binary-tree/discuss/142430/My-Python-Solution:-how-do-you-think-about-it)
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
            root.left = find_root(l)
            root.right = find_root(r)
            return root
        return find_root(nums)
```
  
## 832. Flipping an Image: [(original link)](https://leetcode.com/problems/flipping-an-image/description/)  
Given a binary matrix A, we want to flip the image horizontally, then invert it, and return the resulting image.

To flip an image horizontally means that each row of the image is reversed.  For example, flipping [1, 1, 0] horizontally results in [0, 1, 1].

To invert an image means that each 0 is replaced by 1, and each 1 is replaced by 0. For example, inverting [0, 1, 1] results in [1, 0, 0].

>Example 1:  
Input: [[1,1,0],[1,0,1],[0,0,0]]  
Output: [[1,0,0],[0,1,0],[1,1,1]]  
Explanation: First reverse each row: [[0,1,1],[1,0,1],[0,0,0]].  
Then, invert the image: [[1,0,0],[0,1,0],[1,1,1]]  
  
## Analysis  
This is a pretty easy question. The objective to post it here is to show how to use python's simple syntax to implement a "one-line" soution.  
## Solution  
```
class Solution(object):
    def flipAndInvertImage(self, A):
        """
        :type A: List[List[int]]
        :rtype: List[List[int]]
        """
        return [[(1-i) for i in j[::-1]] for j in A]
            
```
  
## 461. Hamming Distance [original link](https://leetcode.com/problems/hamming-distance/description/)  
The Hamming distance between two integers is the number of positions at which the corresponding bits are different.

Given two integers **x** and **y**, calculate the Hamming distance.

Note:
0 ≤ x, y < 2^31.  
>Example:  
Input: x = 1, y = 4  
Output: 2  
Explanation:  
1 (0 0 0 1)  
4 (0 1 0 0)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↑&nbsp;&nbsp;&nbsp;&nbsp;↑   
The above arrows point to positions where the corresponding bits are different   
  
## Solution  
Here we will use the python built-in function ```bin```, which will convert an integer number to a binary string prefixed with “0b”.  
```
class Solution(object):
    def hammingDistance(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        dx = list(bin(x)[2:])
        dy = list(bin(y)[2:])
        while not len(dx) == len(dy):
            if len(dx) < len(dy):
                dx.insert(0,'0')
            else:
                dy.insert(0,'0')
        result = 0
        for i in range(len(dx)):
            if dx[i]!= dy[i]:
                result += 1 
        return result
```
Another tricky 1-line [answer](https://leetcode.com/problems/hamming-distance/discuss/143949/One-line-python-code)   
```class Solution(object):
    def hammingDistance(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        return bin(x ^ y)[2:].count('1')
```  
In this solution, we used python's bitwise XOR operator '^' to compute the the hamming distance between x and y.  
  
## 6. ZigZag Conversion [original link](https://leetcode.com/problems/zigzag-conversion/description/)  
The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this:  
P&emsp;A&emsp;H&emsp;N  
A&thinsp;P&thinsp;L&thinsp;&thinsp;S&ensp;I&ensp;I&ensp;G  
Y&emsp;I&emsp;&thinsp;&thinsp;R   
  
And then read line by line: "PAHNAPLSIIGYIR"  

Write the code that will take a string and make this conversion given a number of rows:  

`string convert(string s, int numRows);`  
>Example 1:

Input: s = "PAYPALISHIRING", numRows = 3
Output: "PAHNAPLSIIGYIR"
  
## Solution 1: (My original method)Construct a 2-d array that stores all the characters. The "ZigZag" part will be stored  along with "".   
```
class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if len(s) <= 2 or numRows==1:
            return s
        n = len(s)/(2*numRows-2) + 1
        ZZ = []
        k = 0
        for i in range(n):
            try:
                ZZ.append([s[p] for p in range(k, k+numRows)])
            except: 
                temp = ['' for t in range(numRows)]
                for p in range(0, len(s)-k):
                    temp[p] = s[k+p]
                T = temp[:]
                ZZ.append(T)
                break
            for j in range(0,numRows-2):
                temp = ['' for t in range(numRows)]
                try:
                    temp[numRows-j-2] = s[k+numRows+j]
                    T = temp[:]
                    ZZ.append(T)
                except:
                    break
            k += numRows+numRows-2
        
        zz = [ZZ[i][j] for j in range(numRows) for i in range(len(ZZ))]
        result = ''.join(zz)
        return result    
```
Drawback: need to decide the numeber of the columns in advance; in each column we have to create a array with the same length, with a redundancy of ```numRows - 1``` characters of "".  
## Solution 2: variable  lengths' 2-d array.   
In this solution, we introduce a `step` variable, which indicates the direction of Zig-Zag. `step = 1` while counting from top to bottom, and `step = -1` while counting from bottom to top. The difference with the previous method is that the 2d array is not a rectangular matrix. The length of each dimension is different, which avoid inserting many "" into the array.
```
class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if numRows == 1:
            return s
        s_zig = [[] for i in range(numRows)]
        step = 1
        idx = 0
        for i in range(len(s)):
            s_zig[idx].append(s[i])
            if idx == 0:
                step = 1
            elif idx == numRows-1:
                step = -1
            idx += step
        return ''.join([''.join(s_zig[i]) for i in range(numRows)])
```
  

**Solution1: merge two arrays**  

The most straightforward method we might come up with is to merge the two arrays and find the median of the merged arrays. Inspired by merge sort algorithm, we can easily write the `merge()` function to do this as follows:
```
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        def merge(n1,n2):
            result = []
            i = 0
            j = 0
            while i < len(n1) and j < len(n2):
                if n1[i] < n2[j]:
                    result.append(n1[i])
                    i += 1
                else:
                    result.append(n2[j])
                    j += 1
            result += n1[i:]
            result += n2[j:]
            return result
        nums = merge(nums1, nums2)
        if len(nums) <= 1:
            return float(nums[0])
        if len(nums)%2 == 0:
            mid = len(nums)//2
            return (nums[mid-1] + nums[mid])/2.0
        else:
            mid = len(nums)//2
            return float(nums[mid])
```
However, the time complexity of this solution is **O(m+n)** since we traverse the two entire arrays. To achieve the complexity of **O(log(m+n))**, we can adopt the following solution(inspired by [windeliang](http://windliang.cc/2018/07/18/leetCode-4-Median-of-Two-Sorted-Arrays/)).
  
**Solution 2: binary search**  
This problem is actually a variation of **<<finding the kth smallest/largest element>>**, and the difference is that we have two separate arrays now. We  can adopt binary search method for this.
  
Supposing `L` is the length of the merged array, so median is the `(L/2+1)`th  element(for odd number of elements) or the mean of `(L/2)`th and `(L/2+1)`th elements(for even number of elements).
  
Let `k` = `L/2`, now the key is to find the `k`th element. We don't need to really merge the two arrays. Instead, we can compare the `k/2`th elements in both lists `nums1` and `nums2`. If the `k/2`th elements in `nums1` is smaller than in `nums2`, we know that the first `k/2` elements in `nums1` cannot be the median number and thus we remove them, as indicated in the figure below.![Image of Yaktocat](https://github.com/ChenWentai/LeetCode-problems/blob/master/images/problem4_MergeSortedArray.jpg)  
  
In this example, `k = 7`, and the 3th element in `nums2` is smaller, so we remove the first 3 elements in `nums2` and started this process again. Next time `L = (14-3) = 11`  and `k = 1/2 = 5` so we just need to find the 5th element. Again `k/2 = 2` so we remove the first 2 elements in one array. This iteration continues untill `k == 1` or one of the array becomes empty, and the result(**median**) is the **0 th** element in the current two arrays(smaller one) or `k`th element in the only array(another array is empty). The code is shown below:  
```
class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        def getKth(nums1, nums2, k):
            # let nums1 to be the shorter array 
            if len(nums1) > len(nums2):
                return getKth(nums2, nums1, k)
            if len(nums1) == 0:
                return (nums2[k-1])
            if k == 1:
                return min(nums1[0], nums2[0])
            mid1 = int(min(k/2, len(nums1)))
            mid2 = int(min(k/2, len(nums2)))
            # print("mid1:",mid1, "mid2:",mid2)
            if nums1[mid1-1] > nums2[mid2-1]:
                k = k - mid2
                return getKth(nums1, nums2[mid2:],k)
            else:
                k = k - mid1
                return getKth(nums1[mid1:], nums2, k)
        # merge the even case and odd case. If the number of elements is odd, the getKth() can still return the same result with a different k.
        k1 = (len(nums1) + len(nums2)+1) // 2
        k2 = (len(nums1) + len(nums2)+2) // 2
        return (getKth(nums1, nums2, k1) + getKth(nums1, nums2, k2))/2.0
        ```

  
