from rouge_score import rouge_scorer
class ROUGE():
    
    def rouge(self,target_list,predict_list,func='sum',type_score='rouge1'):
        precision=[]
        recall=[]
        f1=[]
        scorer1 = rouge_scorer.RougeScorer([type_score])
        for target,pred in zip(target_list,predict_list):
            rouge=scorer1.score(target,pred)[type_score]
            precision.append(rouge.precision)
            recall.append(rouge.recall)
            f1.append(rouge.fmeasure)
        if func=='sum':
            return {
            'precision':sum(precision),
            'recall':sum(recall),
            'f1':sum(f1)            
        }
        elif func=='avg':
            return {
            'precision':sum(precision)/len(precision),
            'recall':sum(recall)/len(recall),
            'f1':sum(f1)/len(f1)            
        }
        else:
            return {
            'precision':precision,
            'recall':recall,
            'f1':f1            
        }

    def rougeall(self,target_list,predict_list,func='sum'):
        res={}
        scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'])
        
        return res
        
    
