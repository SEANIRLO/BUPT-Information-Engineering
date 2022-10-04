package com.example.bburecog.utils;

import ohos.media.image.ImageSource;
import ohos.media.image.PixelMap;

import java.io.IOException;
import java.io.InputStream;
import java.util.Base64;

public class Base64Util {
    /**
     * Encode imageInputStream to Base64 String
     */
    public static String imageInputStreamToBase64(InputStream is){
        if(is==null){
            return null;
        }
        byte[] data = null;
        String result = null;
        try{
            data = new byte[is.available()];
            is.read(data);
            result = Base64.getEncoder().encodeToString(data);
        }catch (Exception e){
            e.printStackTrace();
        }finally {
            try {
                is.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return result;
    }

    /**
     * Decode Base64 String to PixelMap
     */
    public static PixelMap base64ToPixelMap(String Base64Data){
        byte[] base64byte=     java.util.Base64.getDecoder().decode(Base64Data);
        ImageSource.SourceOptions srcOpts = new ImageSource.SourceOptions();
        srcOpts.formatHint = "image/png";
        ImageSource imageSource = ImageSource.create(base64byte, srcOpts);
        ImageSource.DecodingOptions decodingOptions = new ImageSource.DecodingOptions();
        return imageSource.createPixelmap(decodingOptions);
    }
}