// pushthis with a readme file
// create a jar file for this and push
// create a jar file for v2 and push
// in a readme file add this
/*
 * Copyright (c) 2019 Automation Anywhere.
 * All rights reserved.
 *
 * This software is the proprietary information of Automation Anywhere.
 * You shall use it only in accordance with the terms of the license agreement
 * you entered into with Automation Anywhere.
 */
/**
 *
 */
package com.automationanywhere.botcommand.samples.commands.basic;

import com.automationanywhere.botcommand.data.Value;
import com.automationanywhere.botcommand.data.impl.StringValue;
import com.automationanywhere.commandsdk.annotations.BotCommand;
import com.automationanywhere.commandsdk.annotations.CommandPkg;
import com.automationanywhere.commandsdk.annotations.Execute;
import com.automationanywhere.commandsdk.annotations.Idx;
import com.automationanywhere.commandsdk.annotations.Pkg;
import com.automationanywhere.commandsdk.annotations.rules.NotEmpty;
import com.automationanywhere.commandsdk.i18n.Messages;
import com.automationanywhere.commandsdk.i18n.MessagesFactory;
import com.automationanywhere.core.security.SecureString;

import java.io.UnsupportedEncodingException;
import java.security.AlgorithmParameters;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.security.spec.InvalidKeySpecException;
import java.security.spec.InvalidParameterSpecException;
import javax.crypto.BadPaddingException;
import javax.crypto.Cipher;
import javax.crypto.IllegalBlockSizeException;
import javax.crypto.NoSuchPaddingException;
import javax.crypto.SecretKey;
import javax.crypto.SecretKeyFactory;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.PBEKeySpec;
import javax.crypto.spec.SecretKeySpec;
import org.apache.commons.codec.binary.Base64;


import static com.automationanywhere.commandsdk.model.AttributeType.CREDENTIAL;
import static com.automationanywhere.commandsdk.model.AttributeType.TEXT;
import static com.automationanywhere.commandsdk.model.DataType.STRING;



//BotCommand makes a class eligible for being considered as an action.
@BotCommand

//CommandPks adds required information to be dispalable on GUI.
@CommandPkg(
		//Unique name inside a package and label to display.
		name = "Encrypt", label = "Encrypt",
		node_label = "Encrypt", description = "Encrypts a string", icon = "pkg.svg",
		
		//Return type information. return_type ensures only the right kind of variable is provided on the UI. 
		return_label = "Assign result to variable", return_type = STRING, return_required = true)
public class Encrypt {
	
	//Messages read from full qualified property file name and provide i18n capability.
	private static final Messages MESSAGES = MessagesFactory
			.getMessages("com.automationanywhere.botcommand.samples.messages");

	//Identify the entry point for the action. Returns a Value<String> because the return type is String. 
	@Execute
	public Value<String> action(
			//Idx 1 would be displayed first, with a text box for entering the value.
			
			
			
			@Idx(index = "1", type = TEXT)
			//UI labels.
			@Pkg(label = "String to encrypt")
			String word,
			
			@Idx(index = "2", type = TEXT) 
			@Pkg(label = "Password")
			@NotEmpty 
			String password) {
				try {
			SecureRandom ex = new SecureRandom();
			byte[] bytes = new byte[20];
			ex.nextBytes(bytes);
			SecretKeyFactory factory = SecretKeyFactory.getInstance("PBKDF2withHmacSHA1");
			PBEKeySpec spec = new PBEKeySpec(password.toCharArray(), bytes, 50, 128);
			SecretKey secretKey = factory.generateSecret(spec);
			SecretKeySpec secret = new SecretKeySpec(secretKey.getEncoded(), "AES");
			Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
			cipher.init(1, secret);
			AlgorithmParameters params = cipher.getParameters();
			byte[] ivBytes = ((IvParameterSpec)params.getParameterSpec(IvParameterSpec.class)).getIV();
			byte[] encryptedTextBytes = cipher.doFinal(word.getBytes("UTF-8"));
			byte[] buffer = new byte[bytes.length + ivBytes.length + encryptedTextBytes.length];
			System.arraycopy(bytes, 0, buffer, 0, bytes.length);
			System.arraycopy(ivBytes, 0, buffer, bytes.length, ivBytes.length);
			System.arraycopy(encryptedTextBytes, 0, buffer, bytes.length + ivBytes.length, encryptedTextBytes.length);
			new Base64();
			return new StringValue((new Base64()).encodeToString(buffer));
		} catch (InvalidParameterSpecException | NoSuchAlgorithmException | InvalidKeySpecException |
				 NoSuchPaddingException | InvalidKeyException | IllegalBlockSizeException |
				 UnsupportedEncodingException | BadPaddingException var15) {
			return new StringValue("ER001: " + var15.getMessage());
		}
		
	}
}
