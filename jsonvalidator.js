/**

 *
 * It handles:
 * - Objects ({}) and arrays ([])
 * - Strings ("") with basic escape sequences (\", \\, \/, \b, \f, \n, \r, \t)
 * - Numbers (integers, decimals, exponents)
 * - Booleans (true, false)
 * - Null (null)
 * - Whitespace handling
 * - Provides more specific error messages for parsing failures.
 
 * @param {string} jsonString The JSON string to validate and parse.
 * @returns {object} An object with:
 * - isValid: boolean (true if valid, false otherwise)
 * - message: string (description of the result or error)
 * - parsedData: any (the parsed JavaScript object/value if valid, null otherwise)
 */
function validateJson(jsonString) {
    if (typeof jsonString !== 'string') {
        return { isValid: false, message: "Input must be a string.", parsedData: null };
    }

    let currentIndex = 0; // Current position in the string during parsing

    /**
     * Skips whitespace characters.
     */
    function skipWhitespace() {
        while (currentIndex < jsonString.length &&
               /\s/.test(jsonString[currentIndex])) {
            currentIndex++;
        }
    }

    /**
     * Parses a string value from the JSON string.
     * Assumes current index is at the opening double quote.
     * @returns {string} The parsed string value.
     * @throws {Error} If string is malformed or unclosed.
     */
    function parseString() {
        // Must start with a double quote
        if (jsonString[currentIndex] !== '"') {
            throw new Error(`Expected '"' at position ${currentIndex} for string start.`);
        }
        currentIndex++; // Move past the opening quote

        let result = '';
        let char;
        let escaped = false;

        while (currentIndex < jsonString.length) {
            char = jsonString[currentIndex];

            if (escaped) {
                // Handle escape sequences
                switch (char) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case '/': result += '/'; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    case 'u': // Unicode escape sequence \uXXXX
                        // This is a simplified handling. A full parser would validate XXXX as hex digits.
                        if (currentIndex + 4 >= jsonString.length) {
                            throw new Error(`Incomplete unicode escape sequence at position ${currentIndex}.`);
                        }
                        const hexCode = jsonString.substring(currentIndex + 1, currentIndex + 5);
                        if (!/^[0-9a-fA-F]{4}$/.test(hexCode)) {
                            throw new Error(`Invalid unicode escape sequence at position ${currentIndex}. Expected 4 hex digits.`);
                        }
                        result += String.fromCharCode(parseInt(hexCode, 16));
                        currentIndex += 4; // Move past the 4 hex digits
                        break;
                    default:
                        // Invalid escape sequence
                        throw new Error(`Invalid escape sequence '\\${char}' at position ${currentIndex - 1}.`);
                }
                escaped = false;
            } else if (char === '\\') {
                escaped = true;
            } else if (char === '"') {
                currentIndex++; // Move past the closing quote
                return result;
            } else {
                result += char;
            }
            currentIndex++;
        }
        throw new Error(`Unterminated string starting at position ${currentIndex - result.length - 1}.`);
    }

    /**
     * Parses a number value from the JSON string.
     * @returns {number} The parsed number value.
     * @throws {Error} If number is malformed.
     */
    function parseNumber() {
        const start = currentIndex;
        // Regex to match JSON numbers: optional minus, digits, optional decimal, optional exponent
        // This regex is a simplified version; full JSON number regex is more complex.
        const numberRegex = /^-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?/;
        const match = jsonString.substring(currentIndex).match(numberRegex);

        if (!match || match.index !== 0) {
            throw new Error(`Invalid number format at position ${currentIndex}.`);
        }

        const numStr = match[0];
        currentIndex += numStr.length;
        
        // Use parseFloat for robust number parsing, as it handles all valid JSON number formats.
        // If we couldn't use parseFloat, we'd have to implement our own number parser, which is very complex.
        const num = parseFloat(numStr);
        if (isNaN(num)) { // Should not happen if regex is correct, but as a safeguard
             throw new Error(`Could not parse number '${numStr}' at position ${start}.`);
        }
        return num;
    }

    /**
     * Parses 'true', 'false', or 'null' keywords.
     * @returns {boolean|null} The parsed boolean or null value.
     * @throws {Error} If keyword is not recognized.
     */
    function parseKeyword() {
        const remaining = jsonString.substring(currentIndex);
        if (remaining.startsWith('true')) {
            currentIndex += 4;
            return true;
        } else if (remaining.startsWith('false')) {
            currentIndex += 5;
            return false;
        } else if (remaining.startsWith('null')) {
            currentIndex += 4;
            return null;
        }
        throw new Error(`Unexpected token at position ${currentIndex}. Expected 'true', 'false', or 'null'.`);
    }

    /**
     * Parses an array from the JSON string.
     * Assumes current index is at the opening square bracket.
     * @returns {Array} The parsed array.
     * @throws {Error} If array is malformed.
     */
    function parseArray() {
        if (jsonString[currentIndex] !== '[') {
            throw new Error(`Expected '[' at position ${currentIndex} for array start.`);
        }
        currentIndex++; // Move past '['
        skipWhitespace();

        const arr = [];
        if (jsonString[currentIndex] === ']') {
            currentIndex++; // Empty array []
            return arr;
        }

        while (currentIndex < jsonString.length) {
            arr.push(parseValue()); // Parse array element
            skipWhitespace();

            if (jsonString[currentIndex] === ',') {
                currentIndex++; // Move past ','
                skipWhitespace();
                if (jsonString[currentIndex] === ']') {
                    throw new Error(`Trailing comma in array at position ${currentIndex - 1}.`);
                }
            } else if (jsonString[currentIndex] === ']') {
                currentIndex++; // Move past ']'
                return arr;
            } else {
                throw new Error(`Expected ',' or ']' in array at position ${currentIndex}.`);
            }
        }
        throw new Error(`Unterminated array starting at position ${currentIndex - 1}.`);
    }

    /**
     * Parses an object from the JSON string.
     * Assumes current index is at the opening curly brace.
     * @returns {object} The parsed object.
     * @throws {Error} If object is malformed.
     */
    function parseObject() {
        if (jsonString[currentIndex] !== '{') {
            throw new Error(`Expected '{' at position ${currentIndex} for object start.`);
        }
        currentIndex++; // Move past '{'
        skipWhitespace();

        const obj = {};
        if (jsonString[currentIndex] === '}') {
            currentIndex++; // Empty object {}
            return obj;
        }

        while (currentIndex < jsonString.length) {
            skipWhitespace();
            const key = parseString(); // Object keys must be strings
            skipWhitespace();

            if (jsonString[currentIndex] !== ':') {
                throw new Error(`Expected ':' after key '${key}' at position ${currentIndex}.`);
            }
            currentIndex++; // Move past ':'
            skipWhitespace();

            const value = parseValue(); // Parse value
            obj[key] = value;
            skipWhitespace();

            if (jsonString[currentIndex] === ',') {
                currentIndex++; // Move past ','
                skipWhitespace();
                if (jsonString[currentIndex] === '}') {
                    throw new Error(`Trailing comma in object at position ${currentIndex - 1}.`);
                }
            } else if (jsonString[currentIndex] === '}') {
                currentIndex++; // Move past '}'
                return obj;
            } else {
                throw new Error(`Expected ',' or '}' in object at position ${currentIndex}.`);
            }
        }
        throw new Error(`Unterminated object starting at position ${currentIndex - 1}.`);
    }

    /**
     * Main function to parse any JSON value (recursive).
     * Determines the type of value based on the current character.
     * @returns {any} The parsed JSON value.
     * @throws {Error} If an unexpected character is encountered.
     */
    function parseValue() {
        skipWhitespace(); // Skip whitespace before determining value type
        const char = jsonString[currentIndex];

        if (char === '{') {
            return parseObject();
        } else if (char === '[') {
            return parseArray();
        } else if (char === '"') {
            return parseString();
        } else if (char === '-' || (char >= '0' && char <= '9')) {
            return parseNumber();
        } else if (char === 't' || char === 'f' || char === 'n') {
            return parseKeyword(); // true, false, null
        } else {
            throw new Error(`Unexpected character '${char}' at position ${currentIndex}.`);
        }
    }

    // --- Main validation logic ---
    try {
        const parsedData = parseValue();
        skipWhitespace(); // Ensure no extra characters after the main JSON value

        if (currentIndex !== jsonString.length) {
            throw new Error(`Unexpected characters after JSON data at position ${currentIndex}.`);
        }

        return {
            isValid: true,
            message: "JSON string is syntactically valid and parsed successfully.",
            parsedData: parsedData
        };
    } catch (error) {
        return {
            isValid: false,
            message: `JSON validation error: ${error.message}`,
            parsedData: null
        };
    }
}

// --- Example Usage ---

console.log("--- Testing with the Complete Manual JSON Validator ---");

// Valid JSON examples
const validJson1 = '{"name": "Alice", "age": 30, "isStudent": true, "grades": [90, 85, 92.5], "address": {"street": "123 Main St", "city": "Anytown"}}';
console.log("\nValid JSON 1:");
let result1 = validateJson(validJson1);
console.log(`  Valid: ${result1.isValid}, Message: ${result1.message}`);
if (result1.isValid) console.log("  Parsed Data:", result1.parsedData);

const validJson2 = '[1, "hello", true, null, {"key": "value"}]';
console.log("\nValid JSON 2:");
let result2 = validateJson(validJson2);
console.log(`  Valid: ${result2.isValid}, Message: ${result2.message}`);
if (result2.isValid) console.log("  Parsed Data:", result2.parsedData);

const validJson3 = '{"message": "This string has an \\"escaped quote\\" and a newline\\ncharacter."}';
console.log("\nValid JSON 3 (escaped chars):");
let result3 = validateJson(validJson3);
console.log(`  Valid: ${result3.isValid}, Message: ${result3.message}`);
if (result3.isValid) console.log("  Parsed Data:", result3.parsedData);

const validJson4 = '{"number": -123.45e+2}';
console.log("\nValid JSON 4 (number with exponent):");
let result4 = validateJson(validJson4);
console.log(`  Valid: ${result4.isValid}, Message: ${result4.message}`);
if (result4.isValid) console.log("  Parsed Data:", result4.parsedData);

const validJson5 = '[]';
console.log("\nValid JSON 5 (empty array):");
let result5 = validateJson(validJson5);
console.log(`  Valid: ${result5.isValid}, Message: ${result5.message}`);
if (result5.isValid) console.log("  Parsed Data:", result5.parsedData);

const validJson6 = '{}';
console.log("\nValid JSON 6 (empty object):");
let result6 = validateJson(validJson6);
console.log(`  Valid: ${result6.isValid}, Message: ${result6.message}`);
if (result6.isValid) console.log("  Parsed Data:", result6.parsedData);

// Invalid JSON examples
const invalidJson1 = '{"name": "Bob", "age": 25,}'; // Trailing comma in object
console.log("\nInvalid JSON 1 (trailing comma in object):");
let invalidResult1 = validateJson(invalidJson1);
console.log(`  Valid: ${invalidResult1.isValid}, Message: ${invalidResult1.message}`);

const invalidJson2 = '[1, 2, 3,]'; // Trailing comma in array
console.log("\nInvalid JSON 2 (trailing comma in array):");
let invalidResult2 = validateJson(invalidJson2);
console.log(`  Valid: ${invalidResult2.isValid}, Message: ${invalidResult2.message}`);

const invalidJson3 = '{"name": "Charlie, "city": "NYC"}'; // Unclosed string
console.log("\nInvalid JSON 3 (unclosed string):");
let invalidResult3 = validateJson(invalidJson3);
console.log(`  Valid: ${invalidResult3.isValid}, Message: ${invalidResult3.message}`);

const invalidJson4 = '{"data": [1, 2, 3}'; // Mismatched bracket/brace
console.log("\nInvalid JSON 4 (mismatched bracket/brace):");
let invalidResult4 = validateJson(invalidJson4);
console.log(`  Valid: ${invalidResult4.isValid}, Message: ${invalidResult4.message}`);

const invalidJson5 = '{"key": value}'; // Unquoted string value
console.log("\nInvalid JSON 5 (unquoted string value):");
let invalidResult5 = validateJson(invalidJson5);
console.log(`  Valid: ${invalidResult5.isValid}, Message: ${invalidResult5.message}`);

const invalidJson6 = '{"key": "value" extra}'; // Extra characters after JSON
console.log("\nInvalid JSON 6 (extra characters):");
let invalidResult6 = validateJson(invalidJson6);
console.log(`  Valid: ${invalidResult6.isValid}, Message: ${invalidResult6.message}`);

const invalidJson7 = '{"num": 1.2.3}'; // Invalid number format
console.log("\nInvalid JSON 7 (invalid number format):");
let invalidResult7 = validateJson(invalidJson7);
console.log(`  Valid: ${invalidResult7.isValid}, Message: ${invalidResult7.message}`);

const invalidJson8 = '{"bool": True}'; // Invalid boolean keyword (capital T)
console.log("\nInvalid JSON 8 (invalid boolean keyword):");
let invalidResult8 = validateJson(invalidJson8);
console.log(`  Valid: ${invalidResult8.isValid}, Message: ${invalidResult8.message}`);

const invalidJson9 = '{"bad_escape": "hello \\x world"}'; // Invalid escape sequence
console.log("\nInvalid JSON 9 (invalid escape sequence):");
let invalidResult9 = validateJson(invalidJson9);
console.log(`  Valid: ${invalidResult9.isValid}, Message: ${invalidResult9.message}`);

const invalidJson10 = '{"incomplete_unicode": "\\u123"}'; // Incomplete unicode escape
console.log("\nInvalid JSON 10 (incomplete unicode escape):");
let invalidResult10 = validateJson(invalidJson10);
console.log(`  Valid: ${invalidResult10.isValid}, Message: ${invalidResult10.message}`);

const invalidJson11 = '{"missing_colon" "value"}'; // Missing colon
console.log("\nInvalid JSON 11 (missing colon):");
let invalidResult11 = validateJson(invalidJson11);
console.log(`  Valid: ${invalidResult11.isValid}, Message: ${invalidResult11.message}`);

const invalidJson12 = 'not json'; // Not starting with { or [
console.log("\nInvalid JSON 12 (not starting with { or [):");
let invalidResult12 = validateJson(invalidJson12);
console.log(`  Valid: ${invalidResult12.isValid}, Message: ${invalidResult12.message}`);
