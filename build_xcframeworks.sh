xcrun xcodebuild -create-xcframework -library tvmruntime/lib/libtvm_runtime.dylib -headers tvmruntime/include -output tvmruntime.xcframework

xcrun xcodebuild -create-xcframework -library sentencepiece/lib/libsentencepiece.0.dylib -headers sentencepiece/include -output sentencepiece.xcframework
