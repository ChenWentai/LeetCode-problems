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
