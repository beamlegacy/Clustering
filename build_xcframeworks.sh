xcrun xcodebuild -create-xcframework -library onnxruntime/lib/libonnxruntime.1.11.1.dylib -headers onnxruntime/include -output onnxruntime.xcframework
xcrun xcodebuild -create-xcframework -library sentencepiece/lib/libsentencepiece.0.dylib -headers sentencepiece/include -output sentencepiece.xcframework
