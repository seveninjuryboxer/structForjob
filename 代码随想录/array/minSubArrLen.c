int minSubArrayLen(int target,int* nums,int numsSize){
    int left=0,right=0;
    int minLength=__INT_MAX__;
    int sum;
    for(;right<numsSize;++right){
        sum=sum+nums[right];
        while(sum>=target){
            int sublength=right-left+1;
            minLength=(minLength<sublength?minLength:sublength);
            
            //there is a unremember fault  xiafang 
            
            sum=sum-nums[left];
            left++;
        }
    }
    return minLength==__INT_MAX__?0:minLength;
}