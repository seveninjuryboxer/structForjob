//a little unfamilar
int* sortedSquares(int* nums,int numsSize,int* returnSize){
    *returnSize=numsSize;
    int left=0,right=numsSize-1;
    int* ans=(int*)malloc(sizeof(int)*numsSize);
    int index=0;
    for(index=numsSize-1;index>=0;index--){
        int lSquare=nums[left]*nums[left];
        int rSquare=nums[right]*nums[right];
        if(lSquare>rSquare){
            ans[index]=lSquare;
            left++;
        }
        else{
            ans[index]=rSquare;
            right--;
        }

    }
    return ans;
}