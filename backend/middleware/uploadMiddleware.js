import multer from "multer";
import path from "path";


const storage = multer.diskStorage({
    destination(req, file, cb) {
        cb(null, "uploads/");
    },
    filename(req, file, cb) {
        const ext=path.extname(file.originalname);
        
        const sessionId=req.params.id || 'unknown';
        cb(null, `${sessionId}-${Date.now()}${ext}`);
    },
}); 

const buildFileFilter = (allowedMimePrefixes, allowedMimeTypes = []) => (req, file, cb) => {
    const isAllowedPrefix = allowedMimePrefixes.some((prefix) => file.mimetype.startsWith(prefix));
    const isAllowedMimeType = allowedMimeTypes.includes(file.mimetype);

    if (isAllowedPrefix || isAllowedMimeType) {
        cb(null, true);
    } else {
        cb(new Error("Unsupported file type"), false);
    }
};

const upload = multer({
    storage: storage,
    fileFilter: buildFileFilter(["audio/"], ["application/octet-stream"]),
    limits: { fileSize: 1024 * 1024 * 10 },
});

const resumeUpload = multer({
    storage: storage,
    fileFilter: buildFileFilter(
        ["application/"],
        [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
            "text/plain",
        ]
    ),
    limits: { fileSize: 1024 * 1024 * 10 },
});

const uploadSingleAudio = upload.single("audioFile");
const uploadSingleResume = resumeUpload.single("resumeFile");

export { uploadSingleAudio, uploadSingleResume };