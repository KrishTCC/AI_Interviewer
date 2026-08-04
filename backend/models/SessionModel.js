import mongoose from "mongoose";

const questionSchema = new mongoose.Schema({
    questionText:{
        type:String,
        required:true
    },
    questionType:{
        type:String,
        enum:["coding","oral"],
        required:true
    },
    idealAnswer:{
        type:String,
        default:"pending"
    },
    topic:{
        type:String,
        default:""
    },
    conceptTags:{
        type:[String],
        default:[]
    },
    sourceContext:{
        type:String,
        default:""
    },
    userAnswerText:{
        type:String,
        default:""
    },
    userSubmittedCode:{
        type:String,
        default:""
    },
    isSubmitted:{
        type:Boolean,
        default:false
    },
    isEvaluated:{
        type:Boolean,
        default:false
    },
    technicalScore:{
        type:Number,
        default:0
    },
    confidenceScore:{
        type:Number,
        default:0
    },
    aiFeedback:{
        type:String,
        default:"Not yet submitted or evaluated"
    },
    semanticSimilarity:{
        type:Number,
        default:0
    },
    missingConcepts:{
        type:[String],
        default:[]
    }
});

const sessionSchema= new mongoose.Schema({
    user: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "User",
        required: true,
        index: true,
    },
    role:{
        type:String,
        required:true
    },
    company:{
        type:String,
        default:""
    },
    interviewMode:{
        type:String,
        enum:["general","company","resume"],
        default:"general"
    },
    level:{
        type:String,
        required:true
    },
    difficulty:{
        type:String,
        default:"Medium"
    },
    interviewType:{
        type:String,
        enum:["oral-only","coding-mix"],
        required:true
    },
    topic:{
        type:String,
        default:""
    },
    resumeSummary:{
        type:String,
        default:""
    },
    status:{
        type:String,
        enum:["pending","in-progress","completed","failed"],
        default:"pending"
    },
    overallScore: {
        type: Number,
        default: 0,
    },
    metrics: {
        avgTechnical: { type: Number, default: 0 },
        avgConfidence: { type: Number, default: 0 },
    },
    questions:[questionSchema],
    startTime:{type:Date,default:Date.now},
    endTime:{type:Date},
   
},{
    timestamps:true
});

const Session = mongoose.model("Session", sessionSchema);
export default Session